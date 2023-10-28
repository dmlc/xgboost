/**
 * Copyright 2023, XGBoost Contributors
 */
#if defined(XGBOOST_USE_NCCL)
#include <cstdint>  // for int8_t, int64_t

#include "../common/cuda_context.cuh"
#include "../common/device_helpers.cuh"
#include "../data/array_interface.h"
#include "allgather.h"  // for AllgatherVOffset
#include "coll.cuh"
#include "comm.cuh"
#include "nccl.h"
#include "xgboost/collective/result.h"  // for Result
#include "xgboost/span.h"               // for Span

namespace xgboost::collective {
Coll* Coll::MakeCUDAVar() { return new NCCLColl{}; }

NCCLColl::~NCCLColl() = default;
namespace {
Result GetNCCLResult(ncclResult_t code) {
  if (code == ncclSuccess) {
    return Success();
  }

  std::stringstream ss;
  ss << "NCCL failure: " << ncclGetErrorString(code) << ".";
  if (code == ncclUnhandledCudaError) {
    // nccl usually preserves the last error so we can get more details.
    auto err = cudaPeekAtLastError();
    ss << "  CUDA error: " << thrust::system_error(err, thrust::cuda_category()).what() << "\n";
  } else if (code == ncclSystemError) {
    ss << "  This might be caused by a network configuration issue. Please consider specifying "
          "the network interface for NCCL via environment variables listed in its reference: "
          "`https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html`.\n";
  }
  return Fail(ss.str());
}

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

[[nodiscard]] Result BitwiseAllReduce(NCCLComm const* pcomm, ncclComm_t handle,
                                      common::Span<std::int8_t> data, Op op) {
  dh::device_vector<std::int8_t> buffer(data.size() * pcomm->World());
  auto* device_buffer = buffer.data().get();

  // First gather data from all the workers.
  CHECK(handle);
  auto rc = GetNCCLResult(
      ncclAllGather(data.data(), device_buffer, data.size(), ncclInt8, handle, pcomm->Stream()));
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
  return Success() << [&] {
    if (IsBitwiseOp(op)) {
      return BitwiseAllReduce(nccl, nccl->Handle(), data, op);
    } else {
      return DispatchDType(type, [=](auto t) {
        using T = decltype(t);
        auto rdata = common::RestoreType<T>(data);
        auto rc = ncclAllReduce(data.data(), data.data(), rdata.size(), GetNCCLType(type),
                                GetNCCLRedOp(op), nccl->Handle(), nccl->Stream());
        return GetNCCLResult(rc);
      });
    }
  } << [&] { return nccl->Block(); };
}

[[nodiscard]] Result NCCLColl::Broadcast(Comm const& comm, common::Span<std::int8_t> data,
                                         std::int32_t root) {
  if (!comm.IsDistributed()) {
    return Success();
  }
  auto nccl = dynamic_cast<NCCLComm const*>(&comm);
  CHECK(nccl);
  return Success() << [&] {
    return GetNCCLResult(ncclBroadcast(data.data(), data.data(), data.size_bytes(), ncclInt8, root,
                                       nccl->Handle(), nccl->Stream()));
  } << [&] { return nccl->Block(); };
}

[[nodiscard]] Result NCCLColl::Allgather(Comm const& comm, common::Span<std::int8_t> data,
                                         std::int64_t size) {
  if (!comm.IsDistributed()) {
    return Success();
  }
  auto nccl = dynamic_cast<NCCLComm const*>(&comm);
  CHECK(nccl);
  auto send = data.subspan(comm.Rank() * size, size);
  return Success() << [&] {
    return GetNCCLResult(
        ncclAllGather(send.data(), data.data(), size, ncclInt8, nccl->Handle(), nccl->Stream()));
  } << [&] { return nccl->Block(); };
}

namespace cuda_impl {
/**
 * @brief Implement allgather-v using broadcast.
 *
 * https://arxiv.org/abs/1812.05964
 */
Result BroadcastAllgatherV(NCCLComm const* comm, common::Span<std::int8_t const> data,
                           common::Span<std::int64_t const> sizes, common::Span<std::int8_t> recv) {
  return Success() << [] { return GetNCCLResult(ncclGroupStart()); } << [&] {
    std::size_t offset = 0;
    for (std::int32_t r = 0; r < comm->World(); ++r) {
      auto as_bytes = sizes[r];
      auto rc = ncclBroadcast(data.data(), recv.subspan(offset, as_bytes).data(), as_bytes,
                              ncclInt8, r, comm->Handle(), dh::DefaultStream());
      if (rc != ncclSuccess) {
        return GetNCCLResult(rc);
      }
      offset += as_bytes;
    }
    return Success();
  } << [] { return GetNCCLResult(ncclGroupEnd()); };
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

  switch (algo) {
    case AllgatherVAlgo::kRing: {
      return Success() << [] { return GetNCCLResult(ncclGroupStart()); } << [&] {
        // get worker offset
        detail::AllgatherVOffset(sizes, recv_segments);
        // copy data
        auto current = recv.subspan(recv_segments[comm.Rank()], data.size_bytes());
        if (current.data() != data.data()) {
          dh::safe_cuda(cudaMemcpyAsync(current.data(), data.data(), current.size_bytes(),
                                        cudaMemcpyDeviceToDevice, nccl->Stream()));
        }
        return detail::RingAllgatherV(comm, sizes, recv_segments, recv);
      } << [] {
        return GetNCCLResult(ncclGroupEnd());
      } << [&] { return nccl->Block(); };
    }
    case AllgatherVAlgo::kBcast: {
      return cuda_impl::BroadcastAllgatherV(nccl, data, sizes, recv);
    }
    default: {
      return Fail("Unknown algorithm for allgather-v");
    }
  }
}
}  // namespace xgboost::collective

#endif  // defined(XGBOOST_USE_NCCL)
