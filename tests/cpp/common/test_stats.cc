#include <gtest/gtest.h>
#include <xgboost/generic_parameters.h>

#include "../../../src/common/stats.h"

namespace xgboost {
namespace common {
TEST(Stats, Percentil) {
  linalg::Tensor<float, 1> arr({21, 0, 15, 50, 40, 0, 35}, {5}, Context::kCpuId);
  std::vector<size_t> index{0, 2, 3, 4, 6};
  auto percentile = Percentile(40.f, Span<size_t const>{index}, arr.HostView());
  std::cout << percentile << std::endl;
}
}  // namespace common
}  // namespace xgboost
