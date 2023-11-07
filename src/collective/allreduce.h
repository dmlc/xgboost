/**
 * Copyright 2023, XGBoost Contributors
 */
#pragma once
#include <cstdint>      // for int8_t
#include <functional>   // for function
#include <type_traits>  // for is_invocable_v, enable_if_t

#include "../common/type.h"             // for EraseType, RestoreType
#include "../data/array_interface.h"    // for ArrayInterfaceHandler
#include "comm.h"                       // for Comm, RestoreType
#include "xgboost/collective/result.h"  // for Result
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

  auto erased_fn = [type, redop](common::Span<std::int8_t const> lhs,
                                 common::Span<std::int8_t> out) {
    CHECK_EQ(lhs.size(), out.size()) << "Invalid input for reduction.";
    auto lhs_t = common::RestoreType<T const>(lhs);
    auto rhs_t = common::RestoreType<T>(out);
    redop(lhs_t, rhs_t);
  };

  return cpu_impl::RingAllreduce(comm, erased, erased_fn, type);
}
}  // namespace xgboost::collective
