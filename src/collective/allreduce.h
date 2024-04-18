/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#pragma once
#include <cstdint>      // for int8_t
#include <functional>   // for function
#include <type_traits>  // for is_invocable_v, enable_if_t
#include <vector>       // for vector

#include "../common/type.h"             // for EraseType, RestoreType
#include "../data/array_interface.h"    // for ToDType, ArrayInterfaceHandler
#include "comm.h"                       // for Comm, RestoreType
#include "comm_group.h"                 // for GlobalCommGroup
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
}  // namespace xgboost::collective
