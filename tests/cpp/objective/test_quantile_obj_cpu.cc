/**
 * Copyright 2024-2026, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>

#include "../helpers.h"
#include "test_quantile_obj.h"
#include "test_regression_obj.h"  // for TestVectorLeafObj

namespace xgboost {
TEST(Objective, DeclareUnifiedTest(Quantile)) {
  Context ctx = MakeCUDACtx(GPUIDX);
  TestQuantile(&ctx);
}

TEST(Objective, DeclareUnifiedTest(QuantileIntercept)) {
  Context ctx = MakeCUDACtx(GPUIDX);
  TestQuantileIntercept(&ctx);
}

TEST(Objective, DeclareUnifiedTest(QuantileVectorLeaf)) {
  Context ctx = MakeCUDACtx(GPUIDX);
  bst_idx_t n_samples = 10;
  std::vector<float> sol_left{1.0f, 4.0f, 7.0f};
  std::vector<float> sol_right{11.0f, 14.0f, 17.0f};
  Args args{{"quantile_alpha", "[0.25, 0.5, 0.75]"}};
  TestVectorLeafObj(&ctx, "reg:quantileerror", args, n_samples, 1u, sol_left, sol_right);
}
}  // namespace xgboost
