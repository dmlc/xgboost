#include <gtest/gtest.h>
#include <xgboost/generic_parameters.h>

#include "../../../src/common/stats.h"

namespace xgboost {
namespace common {
TEST(Stats, Percentil) {
  linalg::Tensor<float, 1> arr({21, 15, 50, 40, 35}, {5}, Context::kCpuId);
  auto percentile = Percentile(40.f, arr.HostView());
  std::cout << percentile << std::endl;
}
}  // namespace common
}  // namespace xgboost
