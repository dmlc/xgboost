/**
 * Copyright 2023, XGBoost Contributors
 */
#pragma once
#include <cstdint>      // for int8_t
#include <type_traits>  // for is_const_v, add_const_t, conditional_t, add_pointer_t

#include "xgboost/span.h"  // for Span
namespace xgboost::common {
template <typename T, typename U = std::conditional_t<std::is_const_v<T>,
                                                      std::add_const_t<std::int8_t>, std::int8_t>>
common::Span<U> EraseType(common::Span<T> data) {
  auto n_total_bytes = data.size_bytes();
  auto erased = common::Span{reinterpret_cast<std::add_pointer_t<U>>(data.data()), n_total_bytes};
  return erased;
}

template <typename T, typename U>
common::Span<T> RestoreType(common::Span<U> data) {
  auto n_total_bytes = data.size_bytes();
  auto restored = common::Span{reinterpret_cast<T*>(data.data()), n_total_bytes / sizeof(T)};
  return restored;
}
}  // namespace xgboost::common
