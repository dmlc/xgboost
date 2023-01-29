/**
 * Copyright 2020-2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>  // Context
#include <xgboost/span.h>

#include "../../../src/common/algorithm.h"

namespace xgboost {
namespace common {
TEST(ArgSort, Basic) {
  Context ctx;
  std::vector<float> inputs {3.0, 2.0, 1.0};
  auto ret = ArgSort<bst_feature_t>(&ctx, inputs.cbegin(), inputs.cend());
  std::vector<bst_feature_t> sol{2, 1, 0};
  ASSERT_EQ(ret, sol);
}
}  // namespace common
}  // namespace xgboost
