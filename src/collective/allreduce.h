/**
 * Copyright 2023-2026, XGBoost Contributors
 */
#pragma once
#include <cstdint>      // for int8_t
#include <cstring>      // for memcpy
#include <functional>   // for function
#include <type_traits>  // for is_invocable_v, enable_if_t
#include <vector>       // for vector

#include "../common/type.h"             // for EraseType, RestoreType
#include "../data/array_interface.h"    // for ToDType, ArrayInterfaceHandler
#include "allgather.h"                  // for AllgatherV
#include "broadcast.h"                  // for Broadcast
#include "comm.h"                       // for Comm, RestoreType
#include "comm_group.h"                 // for GlobalCommGroup
#include "topo.h"                       // for ParentLevel, Parent, Child
#include "xgboost/collective/result.h"  // for Result
#include "xgboost/context.h"            // for Context
#include "xgboost/span.h"               // for Span

namespace xgboost::collective {
namespace cpu_impl {
using Func =
    std::function<void(common::Span<std::int8_t const> lhs, common::Span<std::int8_t> out)>;

Result RingAllreduce(Comm const& comm, common::Span<std::int8_t> data, Func const& op,
                     ArrayInterfaceHandler::Type type);
}  // namespace cpu_impl

template <typename T, typename Fn>
std::enable_if_t<std::is_invocable_v<Fn, common::Span<T const>, common::Span<T>>, Result> Allreduce(
    Comm const& comm, common::Span<T> data, Fn redop) {
  auto erased = common::EraseType(data);
  auto type = ToDType<T>::kType;

  auto erased_fn = [redop](common::Span<std::int8_t const> lhs, common::Span<std::int8_t> out) {
    CHECK_EQ(lhs.size(), out.size()) << "Invalid input for reduction.";
    auto lhs_t = common::RestoreType<T const>(lhs);
    auto rhs_t = common::RestoreType<T>(out);
    redop(lhs_t, rhs_t);
  };

  return cpu_impl::RingAllreduce(comm, erased, erased_fn, type);
}

template <typename T, std::int32_t kDim>
[[nodiscard]] Result Allreduce(Context const* ctx, CommGroup const& comm,
                               linalg::TensorView<T, kDim> data, Op op) {
  if (!comm.IsDistributed()) {
    return Success();
  }
  CHECK(data.Contiguous());
  auto erased = common::EraseType(data.Values());
  auto type = ToDType<T>::kType;

  auto backend = comm.Backend(data.Device());
  return backend->Allreduce(ctx, comm.Ctx(ctx, data.Device()), erased, type, op);
}

template <typename T, std::int32_t kDim>
[[nodiscard]] Result Allreduce(Context const* ctx, linalg::TensorView<T, kDim> data, Op op) {
  return Allreduce(ctx, *GlobalCommGroup(), data, op);
}

/**
 * @brief Specialization for std::vector.
 */
template <typename T, typename Alloc>
[[nodiscard]] Result Allreduce(Context const* ctx, std::vector<T, Alloc>* data, Op op) {
  return Allreduce(ctx, linalg::MakeVec(data->data(), data->size()), op);
}

/**
 * @brief Specialization for scalar value.
 */
template <typename T>
[[nodiscard]] std::enable_if_t<std::is_standard_layout_v<T> && std::is_trivial_v<T>, Result>
Allreduce(Context const* ctx, T* data, Op op) {
  return Allreduce(ctx, linalg::MakeVec(data, 1), op);
}

/**
 * @brief Allreduce a variable-length vector over `comm`.
 *
 * The method performs a tree reduction rooted at rank 0 using `redop`, then broadcasts
 * the result so every rank ends with the same reduced payload in `data`.
 *
 * `redop` must have the signature
 * `void(Fn(const Span<T const>& lhs, const Span<T const>& rhs, std::vector<T>* out))` and must
 * write the combined result into `out`.
 */
template <typename T, typename Fn>
std::enable_if_t<
    std::is_invocable_v<Fn, common::Span<T const>, common::Span<T const>, std::vector<T>*>, Result>
AllreduceV(Comm const& comm, std::vector<T>* data, Fn redop) {
  static_assert(std::is_standard_layout_v<T> && std::is_trivially_copyable_v<T>,
                "AllreduceV supports only standard-layout trivially-copyable types.");
  CHECK(data);
  if (!comm.IsDistributed() || comm.World() == 1) {
    return Success();
  }

  auto const world = comm.World();
  auto const rank = comm.Rank();
  auto constexpr kRoot = 0;

  auto send = [&](std::int32_t peer, std::vector<T> const& vec) {
    std::int64_t n = static_cast<std::int64_t>(vec.size());
    auto n_bytes =
        common::Span<std::int8_t const>{reinterpret_cast<std::int8_t const*>(&n), sizeof(n)};
    return Success() << [&] {
      return comm.Chan(peer)->SendAll(n_bytes);
    } << [&] {
      if (n == 0) {
        return Success();
      }
      auto payload_bytes = static_cast<std::size_t>(n) * sizeof(T);
      auto bytes = common::Span<std::int8_t const>{reinterpret_cast<std::int8_t const*>(vec.data()),
                                                   payload_bytes};
      return comm.Chan(peer)->SendAll(bytes);
    } << [&] {
      return comm.Chan(peer)->Block();
    };
  };

  auto recv = [&](std::int32_t peer, std::vector<T>* out) {
    std::int64_t n = 0;
    auto n_bytes = common::Span<std::int8_t>{reinterpret_cast<std::int8_t*>(&n), sizeof(n)};
    auto rc = Success() << [&] {
      return comm.Chan(peer)->RecvAll(n_bytes);
    } << [&] {
      return comm.Chan(peer)->Block();
    };
    if (!rc.OK()) {
      return rc;
    }
    CHECK_GE(n, 0);
    out->resize(static_cast<std::size_t>(n));
    if (n == 0) {
      return Success();
    }
    auto payload_bytes = static_cast<std::size_t>(n) * sizeof(T);
    auto bytes =
        common::Span<std::int8_t>{reinterpret_cast<std::int8_t*>(out->data()), payload_bytes};
    return Success() << [&] {
      return comm.Chan(peer)->RecvAll(bytes);
    } << [&] {
      return comm.Chan(peer)->Block();
    };
  };

  std::vector<T> incoming;
  std::vector<T> out;
  bool continue_reduce = true;
  for (std::int32_t level = 0; (std::int32_t{1} << level) < world; ++level) {
    if (!continue_reduce) {
      continue;
    }
    if (rank > 0 && binomial_tree::ParentLevel(rank) == level) {
      auto parent = binomial_tree::Parent(rank);
      auto rc = send(parent, *data);
      if (!rc.OK()) {
        return Fail("AllreduceV failed to send data to parent.", std::move(rc));
      }
      continue_reduce = false;
      continue;
    }
    if (binomial_tree::HasChild(rank, level, world)) {
      auto child = binomial_tree::Child(rank, level);
      auto rc = recv(child, &incoming);
      if (!rc.OK()) {
        return Fail("AllreduceV failed to receive data from child.", std::move(rc));
      }
      out.clear();
      redop(common::Span<T const>{data->data(), data->size()},
            common::Span<T const>{incoming.data(), incoming.size()}, &out);
      data->swap(out);
    }
  }

  std::int64_t reduced_size = static_cast<std::int64_t>(rank == kRoot ? data->size() : 0);
  auto rc = Broadcast(comm, common::Span<std::int64_t>{&reduced_size, 1}, kRoot);
  if (!rc.OK()) {
    return Fail("AllreduceV failed to broadcast reduced size.", std::move(rc));
  }
  if (reduced_size == 0) {
    data->clear();
    return Success();
  }
  if (rank != kRoot) {
    data->resize(static_cast<std::size_t>(reduced_size));
  }
  auto reduced = common::Span<T>{data->data(), static_cast<std::size_t>(reduced_size)};
  rc = Broadcast(comm, reduced, kRoot);
  if (!rc.OK()) {
    return Fail("AllreduceV failed to broadcast reduced payload.", std::move(rc));
  }
  return Success();
}

template <typename T, typename Fn>
std::enable_if_t<
    std::is_invocable_v<Fn, common::Span<T const>, common::Span<T const>, std::vector<T>*>, Result>
AllreduceV(Context const* ctx, CommGroup const& comm, std::vector<T>* data, Fn redop) {
  if (!comm.IsDistributed()) {
    return Success();
  }
  auto const& cctx = comm.Ctx(ctx, DeviceOrd::CPU());
  return AllreduceV(cctx, data, redop);
}

template <typename T, typename Fn>
std::enable_if_t<
    std::is_invocable_v<Fn, common::Span<T const>, common::Span<T const>, std::vector<T>*>, Result>
AllreduceV(Context const* ctx, std::vector<T>* data, Fn redop) {
  return AllreduceV(ctx, *GlobalCommGroup(), data, redop);
}
}  // namespace xgboost::collective

#if defined(XGBOOST_USE_NCCL) && defined(__CUDACC__)
#include "../common/cuda_context.cuh"
#include "allreduce_v.cuh"  // for gpu_impl::AllreduceV, AllreduceVScratch

namespace xgboost::collective {
template <typename T>
using AllreduceVScratch = gpu_impl::AllreduceVScratch<T>;

namespace gpu_detail {
template <typename T>
Result CopyDeviceVectorToHost(dh::device_vector<T> const& src, std::vector<T>* dst,
                              cudaStream_t stream) {
  CHECK(dst);
  dst->resize(src.size());
  if (src.empty()) {
    return Success();
  }
  auto rc = GetCUDAResult(cudaMemcpyAsync(dst->data(), src.data().get(), src.size() * sizeof(T),
                                          cudaMemcpyDeviceToHost, stream));
  if (!rc.OK()) {
    return rc;
  }
  return GetCUDAResult(cudaStreamSynchronize(stream));
}

template <typename T>
Result CopyHostVectorToDevice(std::vector<T> const& src, dh::device_vector<T>* dst,
                              cudaStream_t stream) {
  CHECK(dst);
  dst->resize(src.size());
  if (src.empty()) {
    return Success();
  }
  auto rc = GetCUDAResult(cudaMemcpyAsync(dst->data().get(), src.data(), src.size() * sizeof(T),
                                          cudaMemcpyHostToDevice, stream));
  if (!rc.OK()) {
    return rc;
  }
  return GetCUDAResult(cudaStreamSynchronize(stream));
}

template <typename T>
void CopyGatheredSegment(common::Span<std::int8_t const> gathered,
                         std::vector<std::int64_t> const& recv_segments, std::int32_t rank,
                         std::vector<T>* out) {
  CHECK(out);
  CHECK_GE(rank, 0);
  CHECK_LT(static_cast<std::size_t>(rank + 1), recv_segments.size());
  auto begin = recv_segments[rank];
  auto end = recv_segments[rank + 1];
  CHECK_LE(begin, end);
  auto n_bytes = static_cast<std::size_t>(end - begin);
  CHECK_EQ(n_bytes % sizeof(T), 0) << "Invalid gathered segment size.";
  out->resize(n_bytes / sizeof(T));
  if (n_bytes != 0) {
    std::memcpy(out->data(), gathered.data() + begin, n_bytes);
  }
}

template <typename T, typename Fn>
std::enable_if_t<std::is_invocable_v<Fn, dh::device_vector<T> const&, dh::device_vector<T> const&,
                                     dh::device_vector<T>*, cudaStream_t>,
                 Result>
AllreduceVHostFallback(Context const* ctx, CommGroup const& comm, dh::device_vector<T>* data,
                       AllreduceVScratch<T>* scratch, Fn&& redop) {
  CHECK(ctx);
  CHECK(ctx->IsCUDA()) << "GPU AllreduceV requires a CUDA context.";
  CHECK(data);
  CHECK(scratch);

  Context cpu_ctx;
  auto stream = ctx->CUDACtx()->Stream();

  std::vector<T> h_local;
  auto rc = CopyDeviceVectorToHost(*data, &h_local, stream);
  if (!rc.OK()) {
    return Fail("GPU AllreduceV fallback failed to copy local payload to host.", std::move(rc));
  }

  std::vector<std::int64_t> recv_segments;
  HostDeviceVector<std::int8_t> gathered;
  rc = AllgatherV(&cpu_ctx, comm, linalg::MakeVec(h_local.data(), h_local.size()), &recv_segments,
                  &gathered);
  if (!rc.OK()) {
    return Fail("GPU AllreduceV fallback failed to allgather host payloads.", std::move(rc));
  }

  constexpr std::int32_t kRoot = 0;
  std::vector<T> h_result;
  if (comm.Rank() == kRoot) {
    auto gathered_bytes = gathered.ConstHostSpan();
    CopyGatheredSegment(gathered_bytes, recv_segments, kRoot, &h_result);

    rc = CopyHostVectorToDevice(h_result, data, stream);
    if (!rc.OK()) {
      return Fail("GPU AllreduceV fallback failed to stage root payload to device.", std::move(rc));
    }

    std::vector<T> h_peer;
    for (std::int32_t peer = 1; peer < comm.World(); ++peer) {
      CopyGatheredSegment(gathered_bytes, recv_segments, peer, &h_peer);
      rc = CopyHostVectorToDevice(h_peer, &scratch->payload, stream);
      if (!rc.OK()) {
        return Fail("GPU AllreduceV fallback failed to stage peer payload to device.",
                    std::move(rc));
      }
      redop(*data, scratch->payload, &scratch->next, stream);
      std::swap(*data, scratch->next);
    }

    rc = CopyDeviceVectorToHost(*data, &h_result, stream);
    if (!rc.OK()) {
      return Fail("GPU AllreduceV fallback failed to copy reduced payload to host.", std::move(rc));
    }
  }

  std::int64_t reduced_size = comm.Rank() == kRoot ? static_cast<std::int64_t>(h_result.size()) : 0;
  rc = Broadcast(&cpu_ctx, comm, linalg::MakeVec(&reduced_size, 1), kRoot);
  if (!rc.OK()) {
    return Fail("GPU AllreduceV fallback failed to broadcast reduced size.", std::move(rc));
  }

  CHECK_GE(reduced_size, 0);
  if (comm.Rank() != kRoot) {
    h_result.resize(static_cast<std::size_t>(reduced_size));
  }
  if (reduced_size != 0) {
    rc = Broadcast(&cpu_ctx, comm, linalg::MakeVec(h_result.data(), h_result.size()), kRoot);
    if (!rc.OK()) {
      return Fail("GPU AllreduceV fallback failed to broadcast reduced payload.", std::move(rc));
    }
  }

  if (comm.Rank() != kRoot) {
    rc = CopyHostVectorToDevice(h_result, data, stream);
    if (!rc.OK()) {
      return Fail("GPU AllreduceV fallback failed to copy broadcast payload to device.",
                  std::move(rc));
    }
  }
  return Success();
}
}  // namespace gpu_detail

template <typename T, typename Fn>
std::enable_if_t<std::is_invocable_v<Fn, dh::device_vector<T> const&, dh::device_vector<T> const&,
                                     dh::device_vector<T>*, cudaStream_t>,
                 Result>
AllreduceV(Context const* ctx, Comm const& comm, dh::device_vector<T>* data,
           AllreduceVScratch<T>* scratch, Fn&& redop) {
  if (!comm.IsDistributed() || comm.World() == 1) {
    return Success();
  }

  auto nccl = dynamic_cast<NCCLComm const*>(&comm);
  if (nccl == nullptr) {
    return Fail("Distributed GPU AllreduceV requires NCCL support.");
  }

  return gpu_impl::AllreduceV(ctx, *nccl, data, scratch, std::forward<Fn>(redop));
}

template <typename T, typename Fn>
std::enable_if_t<std::is_invocable_v<Fn, dh::device_vector<T> const&, dh::device_vector<T> const&,
                                     dh::device_vector<T>*, cudaStream_t>,
                 Result>
AllreduceV(Context const* ctx, CommGroup const& comm, dh::device_vector<T>* data,
           AllreduceVScratch<T>* scratch, Fn&& redop) {
  CHECK(ctx);
  CHECK(ctx->IsCUDA()) << "GPU AllreduceV requires a CUDA context.";

  if (!comm.IsDistributed()) {
    return Success();
  }

  auto const& cctx = comm.Ctx(ctx, ctx->Device());
  auto nccl = dynamic_cast<NCCLComm const*>(&cctx);
  if (nccl != nullptr) {
    return gpu_impl::AllreduceV(ctx, *nccl, data, scratch, std::forward<Fn>(redop));
  }
  return gpu_detail::AllreduceVHostFallback(ctx, comm, data, scratch, std::forward<Fn>(redop));
}

template <typename T, typename Fn>
std::enable_if_t<std::is_invocable_v<Fn, dh::device_vector<T> const&, dh::device_vector<T> const&,
                                     dh::device_vector<T>*, cudaStream_t>,
                 Result>
AllreduceV(Context const* ctx, dh::device_vector<T>* data, AllreduceVScratch<T>* scratch,
           Fn&& redop) {
  return AllreduceV(ctx, *GlobalCommGroup(), data, scratch, std::forward<Fn>(redop));
}
}  // namespace xgboost::collective
#endif  // defined(XGBOOST_USE_NCCL) && defined(__CUDACC__)
