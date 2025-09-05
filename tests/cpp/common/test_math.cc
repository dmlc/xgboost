/**
 * Copyright 2025, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <numeric>  // for accumulate

#include "../../../src/common/math.h"

namespace xgboost::common {
TEST(Math, Softmax) {
  std::vector<float> values{2.0f, 2.0f, 3.0f, 4.0f};

  Softmax(values.begin(), values.end());
  ASSERT_NEAR(std::accumulate(values.cbegin(), values.cend(), 0.0f), 1.0f, 1e-5f);
}
}  // namespace xgboost::common
