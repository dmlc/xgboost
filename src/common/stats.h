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
inline float Percentile(float percentile, Span<size_t const> index,
                        linalg::TensorView<float const, 1> arr) {
  size_t n = index.size();
  if (n == 0) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  std::vector<size_t> sorted_idx(index.size());
  std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
  std::stable_sort(sorted_idx.begin(), sorted_idx.end(),
                   [&](size_t l, size_t r) { return arr(index(l)) < arr(index(r)); });

  auto val = [&](size_t i) { return arr(index(sorted_idx[i])); };

  if (percentile <= (1 / (n + 1))) {
    return val(0);
  }
  if (percentile >= (n / (n + 1))) {
    return val(sorted_idx.size() - 1);
  }
  double x = percentile * static_cast<double>((n + 1));
  double k = std::floor(x);
  double d = x - k;

  auto v0 = val(static_cast<size_t>(k));
  auto v1 = val(static_cast<size_t>(k) + 1);
  return v0 + d * (v1 - v0);
}
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_STATS_H_
