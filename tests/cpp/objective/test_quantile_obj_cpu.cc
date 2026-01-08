/**
 * Copyright 2024-2026, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>

#include "../helpers.h"
#include "test_quantile_obj.h"

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
  TestQuantileVectorLeaf(&ctx);
}
}  // namespace xgboost
