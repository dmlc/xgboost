/**
 * Copyright 2023-2026, XGBoost Contributors
 */
#pragma once
#include <cstdint>      // for int8_t
#include <functional>   // for function
#include <type_traits>  // for is_invocable_v, enable_if_t
#include <vector>       // for vector

#include "../common/type.h"             // for EraseType, RestoreType
#include "../data/array_interface.h"    // for ToDType, ArrayInterfaceHandler
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
  return backend->Allreduce(comm.Ctx(ctx, data.Device()), erased, type, op);
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
