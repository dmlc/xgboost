/*!
 * Copyright 2022 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_STATS_H_
#define XGBOOST_COMMON_STATS_H_
#include <limits>
#include <vector>

#include "common.h"
#include "xgboost/linalg.h"

namespace xgboost {
namespace common {
/**
 * \brief Percentile with masked array using linear interpolation.
 *
 *   https://www.itl.nist.gov/div898/handbook/prc/section2/prc262.htm
 *
 * \param alpha percentile, must be in range [0, 1].
 * \param index The index of valid elements in arr.
 * \param arr   Input values.
 *
 * \return The result of interpolation.
 */
inline float Percentile(double alpha, Span<size_t const> index,
                        linalg::TensorView<float const, 1> arr) {
  CHECK(alpha >= 0 && alpha <= 1);
  if (index.size() == 0) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  auto n = static_cast<double>(index.size());
  std::vector<size_t> sorted_idx(index.size());
  std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
  std::stable_sort(sorted_idx.begin(), sorted_idx.end(),
                   [&](size_t l, size_t r) { return arr(index(l)) < arr(index(r)); });

  auto val = [&](size_t i) { return arr(index(sorted_idx[i])); };

  if (alpha <= (1 / (n + 1))) {
    return val(0);
  }
  if (alpha >= (n / (n + 1))) {
    return val(sorted_idx.size() - 1);
  }

  double x = alpha * static_cast<double>((n + 1));
  double k = std::floor(x) - 1;
  CHECK_GE(k, 0);
  double d = (x - 1) - k;

  auto v0 = val(static_cast<size_t>(k));
  auto v1 = val(static_cast<size_t>(k) + 1);
  return v0 + d * (v1 - v0);
}
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_STATS_H_
