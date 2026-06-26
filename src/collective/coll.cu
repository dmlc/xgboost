/**
 * Copyright 2023-2025, XGBoost Contributors
 */
#if defined(XGBOOST_USE_NCCL)
#include <chrono>       // for chrono, chrono_literals
#include <cstddef>      // for size_t
#include <cstdint>      // for int8_t, int64_t
#include <functional>   // for bit_and, bit_or, bit_xor
#include <future>       // for future, future_status
#include <memory>       // for shared_ptr
#include <mutex>        // for mutex, unique_lock
#include <string>       // for string
#include <thread>       // for this_thread
#include <type_traits>  // for invoke_result_t, is_same_v, enable_if_t
#include <utility>      // for move

#include "../common/cuda_context.cuh"    // for CUDAContext
#include "../common/cuda_stream.h"       // for StreamRef, Event
#include "../common/device_helpers.cuh"  // for device_vector
#include "../common/threadpool.h"        // for ThreadPool
#include "../common/utils.h"             // for MakeCleanup
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

template <typename Fn, typename R = std::invoke_result_t<Fn, curt::StreamRef>>
[[nodiscard]] std::enable_if_t<std::is_same_v<R, Result>, Result> AsyncLaunch(
    Context const* ctx, common::ThreadPool* pool, NCCLComm const* nccl,
    std::shared_ptr<NcclStub> stub, Fn&& fn) {
  auto stream = nccl->Stream();
  auto user_stream = ctx->CUDACtx()->Stream();

  curt::Event before;
  before.Record(user_stream);
  stream.Wait(before);

  auto user_after = common::MakeCleanup([&] {
    curt::Event ev;
    ev.Record(stream);
    user_stream.Wait(ev);
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
void RunBitwiseAllreduce(curt::StreamRef stream, common::Span<std::int8_t> out_buffer,
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

[[nodiscard]] Result BitwiseAllReduce(Context const* ctx, common::ThreadPool* pool,
                                      NCCLComm const* pcomm, common::Span<std::int8_t> data,
                                      Op op) {
  dh::device_vector<std::int8_t> buffer(data.size() * pcomm->World());
  auto* device_buffer = buffer.data().get();
  auto stub = pcomm->Stub();

  // Outer bracket so the post-allgather reduce kernel (run on the NCCL
  // stream) is synchronised back to the caller's stream.
  return BracketNccl(ctx->CUDACtx()->Stream(), pcomm->Stream(), [&]() -> Result {
    auto rc = AsyncLaunch(ctx, pool, pcomm, stub, [&](curt::StreamRef s) {
      return stub->Allgather(data.data(), device_buffer, data.size(), ncclInt8, pcomm->Handle(), s);
    });
    if (!rc.OK()) {
      return rc;
    }
    // Reduce on the NCCL stream (ordered after the allgather kernel queued
    // by `AsyncLaunch`).
    switch (op) {
      case Op::kBitwiseAND:
        RunBitwiseAllreduce(pcomm->Stream(), data, device_buffer, std::bit_and{}, pcomm->World(),
                            data.size());
        break;
      case Op::kBitwiseOR:
        RunBitwiseAllreduce(pcomm->Stream(), data, device_buffer, std::bit_or{}, pcomm->World(),
                            data.size());
        break;
      case Op::kBitwiseXOR:
        RunBitwiseAllreduce(pcomm->Stream(), data, device_buffer, std::bit_xor{}, pcomm->World(),
                            data.size());
        break;
      default:
        LOG(FATAL) << "Not a bitwise reduce operation.";
    }
    return Success();
  });
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

[[nodiscard]] Result NCCLColl::Allreduce(Context const* ctx, Comm const& comm,
                                         common::Span<std::int8_t> data,
                                         ArrayInterfaceHandler::Type type, Op op) {
  CHECK(ctx);
  if (!comm.IsDistributed()) {
    return Success();
  }
  auto nccl = dynamic_cast<NCCLComm const*>(&comm);
  CHECK(nccl);
  auto stub = nccl->Stub();

  return Success() << [&] {
    if (IsBitwiseOp(op)) {
      return BitwiseAllReduce(ctx, &this->pool_, nccl, data, op);
    } else {
      return DispatchDType(type, [&](auto t) {
        using T = decltype(t);
        auto rdata = common::RestoreType<T>(data);
        return AsyncLaunch(ctx, &this->pool_, nccl, stub, [&](curt::StreamRef s) {
          return stub->Allreduce(data.data(), data.data(), rdata.size(), GetNCCLType(type),
                                 GetNCCLRedOp(op), nccl->Handle(), s);
        });
      });
    }
  } << [&] {
    return nccl->Block();
  };
}

[[nodiscard]] Result NCCLColl::Broadcast(Context const* ctx, Comm const& comm,
                                         common::Span<std::int8_t> data, std::int32_t root) {
  CHECK(ctx);
  if (!comm.IsDistributed()) {
    return Success();
  }
  auto nccl = dynamic_cast<NCCLComm const*>(&comm);
  CHECK(nccl);
  auto stub = nccl->Stub();

  return Success() << [&] {
    return AsyncLaunch(ctx, &this->pool_, nccl, stub, [data, nccl, root, stub](curt::StreamRef s) {
      return stub->Broadcast(data.data(), data.data(), data.size_bytes(), ncclInt8, root,
                             nccl->Handle(), s);
    });
  } << [&] {
    return nccl->Block();
  };
}

[[nodiscard]] Result NCCLColl::Allgather(Context const* ctx, Comm const& comm,
                                         common::Span<std::int8_t> data) {
  CHECK(ctx);
  if (!comm.IsDistributed()) {
    return Success();
  }
  auto nccl = dynamic_cast<NCCLComm const*>(&comm);
  CHECK(nccl);
  auto stub = nccl->Stub();
  auto size = data.size_bytes() / comm.World();

  auto send = data.subspan(comm.Rank() * size, size);
  return Success() << [&] {
    return AsyncLaunch(
        ctx, &this->pool_, nccl, stub, [send, data, size, nccl, stub](curt::StreamRef s) {
          return stub->Allgather(send.data(), data.data(), size, ncclInt8, nccl->Handle(), s);
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
Result BroadcastAllgatherV(NCCLComm const* comm, curt::StreamRef s,
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

[[nodiscard]] Result NCCLColl::AllgatherV(Context const* ctx, Comm const& comm,
                                          common::Span<std::int8_t const> data,
                                          common::Span<std::int64_t const> sizes,
                                          common::Span<std::int64_t> recv_segments,
                                          common::Span<std::int8_t> recv, AllgatherVAlgo algo) {
  CHECK(ctx);
  auto nccl = dynamic_cast<NCCLComm const*>(&comm);
  CHECK(nccl);
  if (!comm.IsDistributed()) {
    return Success();
  }
  auto stub = nccl->Stub();

  switch (algo) {
    case AllgatherVAlgo::kRing: {
      // kRing talks to `NCCLChannel` directly without `AsyncLaunch`; bracket
      // with the caller's stream explicitly.
      return BracketNccl(ctx->CUDACtx()->Stream(), nccl->Stream(), [&] {
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
      });
    }
    case AllgatherVAlgo::kBcast: {
      return AsyncLaunch(ctx, &this->pool_, nccl, stub, [&](curt::StreamRef s) {
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
