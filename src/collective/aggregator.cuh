/**
 * Copyright 2023-2024, XGBoost contributors
 *
 * Higher level functions built on top the Communicator API, taking care of behavioral differences
 * between distributed training and federated collective communication
 * learning.
 */
#pragma once
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "allreduce.h"
#include "xgboost/collective/result.h"  // for Result

namespace xgboost::collective {

/**
 * @brief Find the global sum of the given values across all workers.
 *
 * @tparam T The type of the values.
 *
 * @param values Pointer to the inputs to sum.
 * @param size Number of values to sum.
 */
template <typename T, std::int32_t kDim>
[[nodiscard]] Result GlobalSum(Context const* ctx, linalg::TensorView<T, kDim> values) {
  return collective::Allreduce(ctx, values, collective::Op::kSum);
}
}  // namespace xgboost::collective
