/**
 * Copyright 2023, XGBoost Contributors
 */
#pragma once
#include <cstdint>  // for int32_t, int8_t

#include "comm.h"                       // for Comm
#include "xgboost/collective/result.h"  // for
#include "xgboost/span.h"               // for Span

namespace xgboost::collective {
namespace cpu_impl {
Result Broadcast(Comm const& comm, common::Span<std::int8_t> data, std::int32_t root);
}

/**
 * @brief binomial tree broadcast is used on CPU with the default implementation.
 */
template <typename T>
[[nodiscard]] Result Broadcast(Comm const& comm, common::Span<T> data, std::int32_t root) {
  auto n_total_bytes = data.size_bytes();
  auto erased =
      common::Span<std::int8_t>{reinterpret_cast<std::int8_t*>(data.data()), n_total_bytes};
  return cpu_impl::Broadcast(comm, erased, root);
}
}  // namespace xgboost::collective
