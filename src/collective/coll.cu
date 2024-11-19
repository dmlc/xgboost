/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#if defined(XGBOOST_USE_NCCL)
#include <chrono>       // for chrono, chrono_literals
#include <cstddef>      // for size_t
#include <cstdint>      // for int8_t, int64_t
#include <future>       // for future, future_status
#include <memory>       // for shared_ptr
#include <mutex>        // for mutex, unique_lock
#include <string>       // for string
#include <thread>       // for this_thread
#include <type_traits>  // for invoke_result_t, is_same_v, enable_if_t
#include <utility>      // for move

#include "../common/cleanup.h"           // for Cleanup
#include "../common/device_helpers.cuh"  // for CUDAStreamView, CUDAEvent, device_vector
#include "../common/threadpool.h"        // for ThreadPool
#include "../data/array_interface.h"     // for ArrayInterfaceHandler
#include "allgather.h"                   // for AllgatherVOffset
#include "coll.cuh"                      // for NCCLColl
#include "comm.cuh"                      // for NCCLComm
#include "nccl.h"                        // for ncclHalf, ncclFloat32, ...
#include "nccl_stub.h"                   // for BusyWait
#include "xgboost/collective/result.h"   // for Result, Fail
#include "xgboost/global_config.h"       // for InitNewThread
#include "xgboost/span.h"                // for Span

namespace xgboost::collective {
Coll* Coll::MakeCUDAVar() { return new NCCLColl{}; }

NCCLColl::NCCLColl() : pool_{StringView{"nccl-w"}, 2, InitNewThread{}} {}
NCCLColl::~NCCLColl() = default;

namespace {
auto GetNCCLType(ArrayInterfaceHandler::Type type) {
  auto fatal = [] {
    LOG(FATAL) << "Invalid type for NCCL operation.";
    return ncclHalf;  // dummy return to silent the compiler warning.
  };
  using H = ArrayInterfaceHandler;
  switch (type) {
    case H::kF2:
      return ncclHalf;
    case H::kF4:
      return ncclFloat32;
    case H::kF8:
      return ncclFloat64;
    case H::kF16:
      return fatal();
    case H::kI1:
      return ncclInt8;
    case H::kI2:
      return fatal();
    case H::kI4:
      return ncclInt32;
    case H::kI8:
      return ncclInt64;
    case H::kU1:
      return ncclUint8;
    case H::kU2:
      return fatal();
    case H::kU4:
      return ncclUint32;
    case H::kU8:
      return ncclUint64;
  }
  return fatal();
}

namespace {
struct Chan {
  std::mutex cv_lock;
  std::condition_variable cv;
  // Whether the collective operator is called.
  std::atomic<bool> called{false};

  void Notify() {
    std::unique_lock lock{this->cv_lock};
    this->called = true;
    this->cv.notify_one();
  }
  void WaitFor(std::chrono::seconds timeout) {
    std::unique_lock lock{cv_lock};
    cv.wait_for(lock, timeout, [&] { return static_cast<bool>(this->called); });
  }
};
}  // namespace

template <typename Fn, typename R = std::invoke_result_t<Fn, dh::CUDAStreamView>>
[[nodiscard]] std::enable_if_t<std::is_same_v<R, Result>, Result> AsyncLaunch(
    common::ThreadPool* pool, NCCLComm const* nccl, std::shared_ptr<NcclStub> stub,
    dh::CUDAStreamView stream, Fn&& fn) {
  dh::CUDAEvent e0;
  e0.Record(nccl->Stream());
  stream.Wait(e0);

  auto cleanup = common::MakeCleanup([&] {
    dh::CUDAEvent e1;
    e1.Record(stream);
    nccl->Stream().Wait(e1);
  });

  Chan chan;

  auto busy_wait = [&](ncclResult_t* async_error) {
    using std::chrono_literals::operator""ms;
    do {
      auto rc = GetCUDAResult(stream.Sync(false));
      if (!rc.OK()) {
        return rc;
      }
      // async_error is set to success if abort is called.
      rc = stub->CommGetAsyncError(nccl->Handle(), async_error);
      if (!rc.OK()) {
        return rc;
      }
      if (*async_error == ncclInProgress) {
        std::this_thread::sleep_for(5ms);
      }
    } while (*async_error == ncclInProgress);
    return stub->GetNcclResult(*async_error);
  };

  std::future<Result> fut = pool->Submit([&] {
    ncclResult_t async_error = ncclSuccess;
    return Success() << [&] {
      ncclResult_t async_error;
      auto rc = stub->CommGetAsyncError(nccl->Handle(), &async_error);
      if (!rc.OK()) {
        return rc;
      }
      CHECK_NE(async_error, ncclInProgress);

      rc = fn(stream);

      chan.Notify();

      return rc;
    } << [&] {
      return busy_wait(&async_error);
    } << [&] {
      auto rc = stub->CommGetAsyncError(nccl->Handle(), &async_error);
      if (async_error == ncclInProgress) {
        return Fail("In progress after async wait.", std::move(rc));
      }
      return rc;
    };
  });

  chan.WaitFor(nccl->Timeout());

  auto abort = [&](std::string msg) {
    auto rc = stub->CommAbort(nccl->Handle());
    fut.wait();  // Must block, otherwise the thread might access freed memory.
    return Fail(msg + ": " + std::to_string(nccl->Timeout().count()) + "s.") + std::move(rc);
  };
  if (!chan.called) {
    // Timeout waiting for the NCCL op to return. With older versions of NCCL, the op
    // might block even if the config is set to nonblocking.
    return abort("NCCL future timeout");
  }

  // This actually includes the time for prior kernels due to CUDA async calls.
  switch (fut.wait_for(nccl->Timeout())) {
    case std::future_status::timeout:
      // Timeout waiting for the NCCL op to finish.
      return abort("NCCL timeout");
    case std::future_status::ready:
      return fut.get();
    case std::future_status::deferred:
      return Fail("Invalid future status.");
  }

  return Fail("Unreachable");
}

bool IsBitwiseOp(Op const& op) {
  return op == Op::kBitwiseAND || op == Op::kBitwiseOR || op == Op::kBitwiseXOR;
}

template <typename Func>
void RunBitwiseAllreduce(dh::CUDAStreamView stream, common::Span<std::int8_t> out_buffer,
                         std::int8_t const* device_buffer, Func func, std::int32_t world_size,
                         std::size_t size) {
  dh::LaunchN(size, stream, [=] __device__(std::size_t idx) {
    auto result = device_buffer[idx];
    for (auto rank = 1; rank < world_size; rank++) {
      result = func(result, device_buffer[rank * size + idx]);
    }
    out_buffer[idx] = result;
  });
}

[[nodiscard]] Result BitwiseAllReduce(common::ThreadPool* pool, NCCLComm const* pcomm,
                                      common::Span<std::int8_t> data, Op op,
                                      dh::CUDAStreamView stream) {
  dh::device_vector<std::int8_t> buffer(data.size() * pcomm->World());
  auto* device_buffer = buffer.data().get();
  auto stub = pcomm->Stub();

  // First gather data from all the workers.
  auto rc = AsyncLaunch(pool, pcomm, stub, stream, [&](dh::CUDAStreamView s) {
    return stub->Allgather(data.data(), device_buffer, data.size(), ncclInt8, pcomm->Handle(), s);
  });
  if (!rc.OK()) {
    return rc;
  }

  // Then reduce locally.
  switch (op) {
    case Op::kBitwiseAND:
      RunBitwiseAllreduce(pcomm->Stream(), data, device_buffer, thrust::bit_and<std::int8_t>(),
                          pcomm->World(), data.size());
      break;
    case Op::kBitwiseOR:
      RunBitwiseAllreduce(pcomm->Stream(), data, device_buffer, thrust::bit_or<std::int8_t>(),
                          pcomm->World(), data.size());
      break;
    case Op::kBitwiseXOR:
      RunBitwiseAllreduce(pcomm->Stream(), data, device_buffer, thrust::bit_xor<std::int8_t>(),
                          pcomm->World(), data.size());
      break;
    default:
      LOG(FATAL) << "Not a bitwise reduce operation.";
  }
  return Success();
}

ncclRedOp_t GetNCCLRedOp(Op const& op) {
  ncclRedOp_t result{ncclMax};
  switch (op) {
    case Op::kMax:
      result = ncclMax;
      break;
    case Op::kMin:
      result = ncclMin;
      break;
    case Op::kSum:
      result = ncclSum;
      break;
    default:
      LOG(FATAL) << "Unsupported reduce operation.";
  }
  return result;
}
}  // namespace

[[nodiscard]] Result NCCLColl::Allreduce(Comm const& comm, common::Span<std::int8_t> data,
                                         ArrayInterfaceHandler::Type type, Op op) {
  if (!comm.IsDistributed()) {
    return Success();
  }
  auto nccl = dynamic_cast<NCCLComm const*>(&comm);
  CHECK(nccl);
  auto stub = nccl->Stub();

  return Success() << [&] {
    if (IsBitwiseOp(op)) {
      return BitwiseAllReduce(&this->pool_, nccl, data, op, this->stream_.View());
    } else {
      return DispatchDType(type, [&](auto t) {
        using T = decltype(t);
        auto rdata = common::RestoreType<T>(data);
        return AsyncLaunch(
            &this->pool_, nccl, stub, this->stream_.View(), [&](dh::CUDAStreamView s) {
              return stub->Allreduce(data.data(), data.data(), rdata.size(), GetNCCLType(type),
                                     GetNCCLRedOp(op), nccl->Handle(), s);
            });
      });
    }
  } << [&] {
    return nccl->Block();
  };
}

[[nodiscard]] Result NCCLColl::Broadcast(Comm const& comm, common::Span<std::int8_t> data,
                                         std::int32_t root) {
  if (!comm.IsDistributed()) {
    return Success();
  }
  auto nccl = dynamic_cast<NCCLComm const*>(&comm);
  CHECK(nccl);
  auto stub = nccl->Stub();

  return Success() << [&] {
    return AsyncLaunch(&this->pool_, nccl, stub, this->stream_.View(),
                       [data, nccl, root, stub](dh::CUDAStreamView s) {
                         return stub->Broadcast(data.data(), data.data(), data.size_bytes(),
                                                ncclInt8, root, nccl->Handle(), s);
                       });
  } << [&] {
    return nccl->Block();
  };
}

[[nodiscard]] Result NCCLColl::Allgather(Comm const& comm, common::Span<std::int8_t> data) {
  if (!comm.IsDistributed()) {
    return Success();
  }
  auto nccl = dynamic_cast<NCCLComm const*>(&comm);
  CHECK(nccl);
  auto stub = nccl->Stub();
  auto size = data.size_bytes() / comm.World();

  auto send = data.subspan(comm.Rank() * size, size);
  return Success() << [&] {
    return AsyncLaunch(&this->pool_, nccl, stub, this->stream_.View(),
                       [send, data, size, nccl, stub](dh::CUDAStreamView s) {
                         return stub->Allgather(send.data(), data.data(), size, ncclInt8,
                                                nccl->Handle(), s);
                       });
  } << [&] {
    return nccl->Block();
  };
}

namespace cuda_impl {
/**
 * @brief Implement allgather-v using broadcast.
 *
 * https://arxiv.org/abs/1812.05964
 */
Result BroadcastAllgatherV(NCCLComm const* comm, dh::CUDAStreamView s,
                           common::Span<std::int8_t const> data,
                           common::Span<std::int64_t const> sizes, common::Span<std::int8_t> recv) {
  auto stub = comm->Stub();
  return Success() << [&stub] {
    return stub->GroupStart();
  } << [&] {
    std::size_t offset = 0;
    for (std::int32_t r = 0; r < comm->World(); ++r) {
      auto as_bytes = sizes[r];
      auto rc = stub->Broadcast(data.data(), recv.subspan(offset, as_bytes).data(), as_bytes,
                                ncclInt8, r, comm->Handle(), s);
      if (!rc.OK()) {
        return rc;
      }
      offset += as_bytes;
    }
    return Success();
  } << [&] {
    return stub->GroupEnd();
  };
}
}  // namespace cuda_impl

[[nodiscard]] Result NCCLColl::AllgatherV(Comm const& comm, common::Span<std::int8_t const> data,
                                          common::Span<std::int64_t const> sizes,
                                          common::Span<std::int64_t> recv_segments,
                                          common::Span<std::int8_t> recv, AllgatherVAlgo algo) {
  auto nccl = dynamic_cast<NCCLComm const*>(&comm);
  CHECK(nccl);
  if (!comm.IsDistributed()) {
    return Success();
  }
  auto stub = nccl->Stub();

  switch (algo) {
    case AllgatherVAlgo::kRing: {
      return Success() << [&] {
        return stub->GroupStart();
      } << [&] {
        // get worker offset
        detail::AllgatherVOffset(sizes, recv_segments);
        // copy data
        auto current = recv.subspan(recv_segments[comm.Rank()], data.size_bytes());
        if (current.data() != data.data()) {
          dh::safe_cuda(cudaMemcpyAsync(current.data(), data.data(), current.size_bytes(),
                                        cudaMemcpyDeviceToDevice, nccl->Stream()));
        }
        return detail::RingAllgatherV(comm, sizes, recv_segments, recv);
      } << [&] {
        return stub->GroupEnd();
      } << [&] {
        return nccl->Block();
      } << [&] {
        return BusyWait(stub, nccl->Handle(), nccl->Timeout());
      };
    }
    case AllgatherVAlgo::kBcast: {
      return AsyncLaunch(&this->pool_, nccl, stub, this->stream_.View(), [&](dh::CUDAStreamView s) {
        return cuda_impl::BroadcastAllgatherV(nccl, s, data, sizes, recv);
      });
    }
    default: {
      return Fail("Unknown algorithm for allgather-v");
    }
  }
}
}  // namespace xgboost::collective

#endif  // defined(XGBOOST_USE_NCCL)
