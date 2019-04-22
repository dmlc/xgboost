#include <gtest/gtest.h>

#include "../src/rv.h"
#include "../../../tests/cpp/helpers.h"
#include "../../../src/common/random.h"

namespace xgboost {

TEST(RV, Uniform) {
  constexpr float kEps = 1e-4;
  std::vector<float> uniform(10, 0.5f);
  for (size_t i = 0; i < 10; ++i) {
    uniform[i] = Uniform(0, 1).sample();
  }
  for (size_t i = 0; i < 10; ++i) {
    ASSERT_LE(uniform[i], 1.0f);
    ASSERT_GE(uniform[i], 0.0f);
  }

  auto dist = Uniform(0, 1);
  ASSERT_NEAR(dist.mean(), 0.5, kEps);
  ASSERT_NEAR(dist.variance(), 0.08329, kEps);
}

TEST(RV, Normal) {
  constexpr float kEps = 1e-4;
  auto dist = Normal(2.0, 3.0);
  ASSERT_NEAR(dist.mean(), 2.0f, kEps);
  ASSERT_NEAR(dist.variance(), 3.0f, kEps);
}

TEST(RV, Gamma) {
  constexpr float kEps = 1e-4;
  auto dist = Gamma(1.0, 2.0);
  ASSERT_NEAR(dist.mean(), 0.5, kEps);
  ASSERT_NEAR(dist.variance(), 0.25, kEps);
}
}  // namespace xgboost
