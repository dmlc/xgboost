/**
 * Copyright 2020-2023 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_DETERMINISTIC_CUH_
#define XGBOOST_COMMON_DETERMINISTIC_CUH_

#include <cmath>
#include <limits>          // std::numeric_limits

#include "xgboost/base.h"  // XGBOOST_DEVICE

namespace xgboost {
namespace common {
// Following 2 functions are slightly modified version of fbcuda.

/**
 * \brief Constructs a rounding factor used to truncate elements in a sum such that the
 *        sum of the truncated elements is the same no matter what the order of the sum
 *        is.
 *
 * Algorithm 5: Reproducible Sequential Sum in 'Fast Reproducible Floating-Point
 * Summation' by Demmel and Nguyen.
 */
template <typename T>
XGBOOST_DEVICE T CreateRoundingFactor(T max_abs, int n) {
  T delta = max_abs / (static_cast<T>(1.0) -
                       static_cast<T>(2.0) * static_cast<T>(n) * std::numeric_limits<T>::epsilon());

  // Calculate ceil(log_2(delta)).
  // frexpf() calculates exp and returns `x` such that
  // delta = x * 2^exp, where `x` in (-1.0, -0.5] U [0.5, 1).
  // Because |x| < 1, exp is exactly ceil(log_2(delta)).
  int exp;
  std::frexp(delta, &exp);

  // return M = 2 ^ ceil(log_2(delta))
  return std::ldexp(static_cast<T>(1.0), exp);
}

template <typename T>
XGBOOST_DEVICE T TruncateWithRounding(T const rounding_factor, T const x) {
  return (rounding_factor + x) - rounding_factor;
}
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_DETERMINISTIC_CUH_
