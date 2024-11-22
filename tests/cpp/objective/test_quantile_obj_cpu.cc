/**
 * Copyright 2024 by XGBoost contributors
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
}  // namespace xgboost
