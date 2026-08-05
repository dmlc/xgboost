/**
 * Copyright 2023-2024, XGBoost contributors
 *
 * Higher level functions built on top the Communicator API, taking care of behavioral differences
 * between distributed training and federated collective communication
 * learning.
 */
#pragma once
#include <array>
#include <limits>
#include <tuple>
#include <type_traits>

#include "allreduce.h"
#include "xgboost/collective/result.h"  // for Result

namespace xgboost::collective {

/**
 * @brief Find the global max of the given value across all workers.
 */
template <typename T>
std::enable_if_t<std::is_trivially_copy_assignable_v<T>, T> GlobalMax(Context const* ctx, T value) {
  auto rc = collective::Allreduce(ctx, linalg::MakeVec(&value, 1), collective::Op::kMax);
  SafeColl(rc);
  return value;
}

/**
 * @brief Find the global sum of the given values across all workers.
 */
template <typename T, std::int32_t kDim>
[[nodiscard]] Result GlobalSum(Context const* ctx, linalg::TensorView<T, kDim> values) {
  return collective::Allreduce(ctx, values, collective::Op::kSum);
}

template <typename T>
[[nodiscard]] Result GlobalSum(Context const* ctx, linalg::VectorView<T> values,
                               double* sum_weight) {
  auto status = Success() << [&] {
    return Allreduce(ctx, sum_weight, collective::Op::kSum);
  } << [&] {
    return Allreduce(ctx, values, collective::Op::kSum);
  };
  return status;
}

/**
 * @brief Find the global ratio of the given two values across all workers.
 */
template <typename T>
T GlobalRatio(Context const* ctx, T dividend, T divisor) {
  std::array<T, 2> results{dividend, divisor};
  auto rc = GlobalSum(ctx, linalg::MakeVec(results.data(), results.size()));
  SafeColl(rc);
  std::tie(dividend, divisor) = std::tuple_cat(results);
  if (divisor <= 0) {
    return std::numeric_limits<T>::quiet_NaN();
  } else {
    return dividend / divisor;
  }
}
}  // namespace xgboost::collective
