/**
 * Copyright 2020-2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>  // Context
#include <xgboost/span.h>

#include <algorithm>  // is_sorted

#include "../../../src/common/algorithm.h"

namespace xgboost {
namespace common {
TEST(Algorithm, ArgSort) {
  Context ctx;
  std::vector<float> inputs{3.0, 2.0, 1.0};
  auto ret = ArgSort<bst_feature_t>(&ctx, inputs.cbegin(), inputs.cend());
  std::vector<bst_feature_t> sol{2, 1, 0};
  ASSERT_EQ(ret, sol);
}

TEST(Algorithm, Sort) {
  Context ctx;
  ctx.Init(Args{{"nthread", "8"}});
  std::vector<float> inputs{3.0, 1.0, 2.0};

  Sort(&ctx, inputs.begin(), inputs.end(), std::less<>{});
  ASSERT_TRUE(std::is_sorted(inputs.cbegin(), inputs.cend()));

  inputs = {3.0, 1.0, 2.0};
  StableSort(&ctx, inputs.begin(), inputs.end(), std::less<>{});
  ASSERT_TRUE(std::is_sorted(inputs.cbegin(), inputs.cend()));
}
}  // namespace common
}  // namespace xgboost
