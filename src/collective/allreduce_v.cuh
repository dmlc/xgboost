/**
 * Copyright 2026, XGBoost Contributors
 */
#pragma once

#if defined(XGBOOST_USE_NCCL)
#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>
#include <utility>

#include "../common/device_helpers.cuh"  // for device_vector
#include "comm.cuh"                      // for NCCLComm, GetCUDAResult
#include "nccl_stub.h"                   // for BusyWait
#include "topo.h"                        // for binomial tree helpers
#include "xgboost/collective/result.h"
#include "xgboost/logging.h"

namespace xgboost::collective::gpu_impl {
template <typename T>
struct AllreduceVScratch {
  dh::device_vector<T> payload;
  dh::device_vector<T> next;
  dh::device_vector<std::int64_t> size;

  void Reserve(std::size_t n) {
    payload.reserve(n);
    next.reserve(n);
    size.resize(1);
  }
};

template <typename T, typename Fn>
std::enable_if_t<
    std::is_invocable_v<Fn, dh::device_vector<T> const&, dh::device_vector<T> const&,
                        dh::device_vector<T>*, cudaStream_t>,
    Result>
AllreduceV(NCCLComm const& nccl, dh::device_vector<T>* data, AllreduceVScratch<T>* scratch,
           Fn&& redop) {
  static_assert(std::is_standard_layout_v<T> && std::is_trivially_copyable_v<T>,
                "AllreduceV requires trivially-copyable payload elements.");
  CHECK(data);
  CHECK(scratch);

  if (nccl.World() == 1) {
    return Success();
  }
  if (scratch->size.empty()) {
    scratch->size.resize(1);
  }

  auto stub = nccl.Stub();
  auto stream = cudaStream_t{nccl.Stream()};
  auto wait_nccl = [&] {
    auto rc = BusyWait(stub, nccl.Handle(), nccl.Timeout());
    if (!rc.OK()) {
      return rc;
    }
    return GetCUDAResult(cudaStreamSynchronize(stream));
  };

  auto send_size = [&](std::int32_t peer, std::int64_t n) {
    auto rc = GetCUDAResult(cudaMemcpyAsync(scratch->size.data().get(), &n, sizeof(n),
                                            cudaMemcpyHostToDevice, stream));
    if (!rc.OK()) {
      return rc;
    }
    rc = stub->Send(scratch->size.data().get(), 1, ncclInt64, peer, nccl.Handle(), stream);
    if (!rc.OK()) {
      return rc;
    }
    return wait_nccl();
  };

  auto recv_size = [&](std::int32_t peer, std::int64_t* n) {
    CHECK(n);
    auto rc = stub->Recv(scratch->size.data().get(), 1, ncclInt64, peer, nccl.Handle(), stream);
    if (!rc.OK()) {
      return rc;
    }
    rc = wait_nccl();
    if (!rc.OK()) {
      return rc;
    }
    rc = GetCUDAResult(cudaMemcpyAsync(n, scratch->size.data().get(), sizeof(*n),
                                       cudaMemcpyDeviceToHost, stream));
    if (!rc.OK()) {
      return rc;
    }
    return GetCUDAResult(cudaStreamSynchronize(stream));
  };

  auto send_vec = [&](std::int32_t peer, dh::device_vector<T> const& payload) {
    auto rc = send_size(peer, static_cast<std::int64_t>(payload.size()));
    if (!rc.OK() || payload.empty()) {
      return rc;
    }

    auto count = payload.size() * sizeof(T);
    rc = stub->Send(reinterpret_cast<std::int8_t const*>(payload.data().get()), count, ncclInt8,
                    peer, nccl.Handle(), stream);
    if (!rc.OK()) {
      return rc;
    }
    return wait_nccl();
  };

  auto recv_vec = [&](std::int32_t peer, dh::device_vector<T>* payload) {
    CHECK(payload);
    std::int64_t n = 0;
    auto rc = recv_size(peer, &n);
    if (!rc.OK()) {
      return rc;
    }
    CHECK_GE(n, 0);
    payload->resize(static_cast<std::size_t>(n));
    if (n == 0) {
      return Success();
    }

    auto count = static_cast<std::size_t>(n) * sizeof(T);
    rc = stub->Recv(reinterpret_cast<std::int8_t*>(payload->data().get()), count, ncclInt8, peer,
                    nccl.Handle(), stream);
    if (!rc.OK()) {
      return rc;
    }
    return wait_nccl();
  };

  auto rank = nccl.Rank();
  auto world = nccl.World();

  bool continue_reduce = true;
  for (std::int32_t level = 0; (std::int32_t{1} << level) < world; ++level) {
    if (!continue_reduce) {
      continue;
    }
    if (rank > 0 && binomial_tree::ParentLevel(rank) == level) {
      auto parent = binomial_tree::Parent(rank);
      auto rc = send_vec(parent, *data);
      if (!rc.OK()) {
        return Fail("AllreduceV failed to send payload to parent.", std::move(rc));
      }
      continue_reduce = false;
      continue;
    }
    if (binomial_tree::HasChild(rank, level, world)) {
      auto child = binomial_tree::Child(rank, level);
      auto rc = recv_vec(child, &scratch->payload);
      if (!rc.OK()) {
        return Fail("AllreduceV failed to receive payload from child.", std::move(rc));
      }
      redop(*data, scratch->payload, &scratch->next, stream);
      std::swap(*data, scratch->next);
    }
  }

  constexpr std::int32_t kRoot = 0;
  std::int64_t n = 0;
  if (rank == kRoot) {
    n = static_cast<std::int64_t>(data->size());
    auto rc = GetCUDAResult(cudaMemcpyAsync(scratch->size.data().get(), &n, sizeof(n),
                                            cudaMemcpyHostToDevice, stream));
    if (!rc.OK()) {
      return rc;
    }
  }

  auto rc = stub->Broadcast(scratch->size.data().get(), scratch->size.data().get(), 1, ncclInt64,
                            kRoot, nccl.Handle(), stream);
  if (!rc.OK()) {
    return Fail("AllreduceV failed to broadcast reduced size.", std::move(rc));
  }
  rc = wait_nccl();
  if (!rc.OK()) {
    return rc;
  }
  rc = GetCUDAResult(cudaMemcpyAsync(&n, scratch->size.data().get(), sizeof(n),
                                     cudaMemcpyDeviceToHost, stream));
  if (!rc.OK()) {
    return rc;
  }
  rc = GetCUDAResult(cudaStreamSynchronize(stream));
  if (!rc.OK()) {
    return rc;
  }

  CHECK_GE(n, 0);
  data->resize(static_cast<std::size_t>(n));
  auto count = static_cast<std::size_t>(n) * sizeof(T);
  if (count == 0) {
    return Success();
  }

  rc = stub->Broadcast(reinterpret_cast<std::int8_t const*>(data->data().get()),
                       reinterpret_cast<std::int8_t*>(data->data().get()), count, ncclInt8, kRoot,
                       nccl.Handle(), stream);
  if (!rc.OK()) {
    return Fail("AllreduceV failed to broadcast reduced payload.", std::move(rc));
  }
  return wait_nccl();
}
}  // namespace xgboost::collective::gpu_impl

#endif  // defined(XGBOOST_USE_NCCL)
