/**
 * Copyright 2023-2026, XGBoost Contributors
 */
#pragma once

#include <cstdint>      // for int8_t
#include <type_traits>  // for is_invocable_v, enable_if_t

#include "../common/device_helpers.cuh"  // for device_vector, safe_cuda
#include "../common/cuda_rt_utils.h"     // for DefaultStream
#include "allreduce.h"                   // for Result, Comm

namespace xgboost::collective::gpu_impl {
template <typename T>
[[nodiscard]] Result SendAllDevice(Comm const& comm, std::int32_t peer, T const* ptr,
                                   std::size_t count) {
  return Success() << [&] {
    return comm.Chan(peer)->SendAll(reinterpret_cast<std::int8_t const*>(ptr), count * sizeof(T));
  } << [&] {
    return comm.Chan(peer)->Block();
  };
}

template <typename T>
[[nodiscard]] Result RecvAllDevice(Comm const& comm, std::int32_t peer, T* ptr,
                                   std::size_t count) {
  return Success() << [&] {
    return comm.Chan(peer)->RecvAll(reinterpret_cast<std::int8_t*>(ptr), count * sizeof(T));
  } << [&] {
    return comm.Chan(peer)->Block();
  };
}

template <typename Fn>
std::enable_if_t<std::is_invocable_v<Fn, common::Span<std::int8_t const>,
                                     common::Span<std::int8_t const>,
                                     dh::device_vector<std::int8_t>*>,
                 Result>
AllreduceV(Comm const& comm, dh::device_vector<std::int8_t>* data, Fn redop) {
  CHECK(data);
  if (!comm.IsDistributed() || comm.World() == 1) {
    return Success();
  }

  auto const world = comm.World();
  auto const rank = comm.Rank();
  auto constexpr kRoot = 0;

  auto send = [&](std::int32_t peer, dh::device_vector<std::int8_t> const& vec) {
    dh::device_vector<std::int64_t> d_n_bytes(1);
    std::int64_t n_bytes = static_cast<std::int64_t>(vec.size());
    dh::safe_cuda(cudaMemcpyAsync(d_n_bytes.data().get(), &n_bytes, sizeof(n_bytes),
                                  cudaMemcpyHostToDevice, curt::DefaultStream()));
    dh::safe_cuda(cudaStreamSynchronize(curt::DefaultStream()));
    return Success() << [&] {
      return SendAllDevice(comm, peer, d_n_bytes.data().get(), d_n_bytes.size());
    } << [&] {
      return SendAllDevice(comm, peer, vec.data().get(), vec.size());
    };
  };

  auto recv = [&](std::int32_t peer, dh::device_vector<std::int8_t>* out) {
    dh::device_vector<std::int64_t> d_n_bytes(1);
    return Success() << [&] {
      return RecvAllDevice(comm, peer, d_n_bytes.data().get(), d_n_bytes.size());
    } << [&] {
      std::int64_t n_bytes{0};
      dh::safe_cuda(cudaMemcpyAsync(&n_bytes, d_n_bytes.data().get(), sizeof(n_bytes),
                                    cudaMemcpyDeviceToHost, curt::DefaultStream()));
      dh::safe_cuda(cudaStreamSynchronize(curt::DefaultStream()));
      CHECK_GE(n_bytes, 0);
      out->resize(n_bytes);
      return Success();
    } << [&] {
      return RecvAllDevice(comm, peer, out->data().get(), out->size());
    };
  };

  dh::device_vector<std::int8_t> incoming;
  dh::device_vector<std::int8_t> out;
  bool continue_reduce = true;
  for (std::int32_t level = 0; (std::int32_t{1} << level) < world; ++level) {
    if (!continue_reduce) {
      continue;
    }
    if (rank > 0 && binomial_tree::ParentLevel(rank) == level) {
      auto parent = binomial_tree::Parent(rank);
      auto rc = send(parent, *data);
      if (!rc.OK()) {
        return Fail("GPU AllreduceV failed to send data to parent.", std::move(rc));
      }
      continue_reduce = false;
      continue;
    }
    if (binomial_tree::HasChild(rank, level, world)) {
      auto child = binomial_tree::Child(rank, level);
      auto rc = recv(child, &incoming);
      if (!rc.OK()) {
        return Fail("GPU AllreduceV failed to receive data from child.", std::move(rc));
      }
      redop(common::Span<std::int8_t const>{data->data().get(), data->size()},
            common::Span<std::int8_t const>{incoming.data().get(), incoming.size()}, &out);
      *data = std::move(out);
    }
  }

  if (rank != kRoot) {
    auto parent = binomial_tree::Parent(rank);
    auto rc = recv(parent, data);
    if (!rc.OK()) {
      return Fail("GPU AllreduceV failed to receive broadcast data from parent.", std::move(rc));
    }
  }

  for (std::int32_t level = binomial_tree::Depth(world); level >= 0; --level) {
    if (binomial_tree::HasChild(rank, level, world)) {
      auto child = binomial_tree::Child(rank, level);
      auto rc = send(child, *data);
      if (!rc.OK()) {
        return Fail("GPU AllreduceV failed to send broadcast data to child.", std::move(rc));
      }
    }
  }

  return Success();
}
}  // namespace xgboost::collective::gpu_impl
