#include <limits>
#include <vector>

#include "common.h"
#include "xgboost/linalg.h"

namespace xgboost {
namespace common {
float Percentile(float percentile, linalg::TensorView<float const, 1> arr) {
  size_t n = arr.Shape(0);
  if (n == 0) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  std::vector<size_t> sorted_idx{ArgSort(arr)};

  if (percentile <= (1 / (n + 1))) {
    return arr(sorted_idx.front());
  }
  if (percentile >= (n / (n + 1))) {
    return arr(sorted_idx.back());
  }
  double x = percentile * static_cast<double>((n + 1));
  double k = std::floor(x);
  double d = x - k;

  auto v0 = arr(sorted_idx[static_cast<size_t>(k)]);
  auto v1 = arr(sorted_idx[static_cast<size_t>(k) + 1]);
  return v0 + d * (v1 - v0);
}
}  // namespace common
}  // namespace xgboost
