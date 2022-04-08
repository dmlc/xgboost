#include <gtest/gtest.h>
#include <xgboost/generic_parameters.h>

#include "../../../src/common/stats.h"

namespace xgboost {
namespace common {
TEST(Stats, Percentil) {
  linalg::Tensor<float, 1> arr({20.f, 0.f, 15.f, 50.f, 40.f, 0.f, 35.f}, {7}, Context::kCpuId);
  std::vector<size_t> index{0, 2, 3, 4, 6};
  // auto percentile = Percentile(0.40f, Span<size_t const>{index}, arr.HostView());
  // ASSERT_EQ(percentile, 26.0);

  // percentile = Percentile(0.20f, Span<size_t const>{index}, arr.HostView());
  // ASSERT_EQ(percentile, 16.0);

  // percentile = Percentile(0.10f, Span<size_t const>{index}, arr.HostView());
  // ASSERT_EQ(percentile, 15.0);
}
}  // namespace common
}  // namespace xgboost
