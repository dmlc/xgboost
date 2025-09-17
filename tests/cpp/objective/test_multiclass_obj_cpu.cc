/*!
 * Copyright 2018-2023 XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>

#include "../helpers.h"
#include "test_multiclass_obj.h"

namespace xgboost {
TEST(Objective, DeclareUnifiedTest(SoftmaxMultiClassObjGPair)) {
  Context ctx = MakeCUDACtx(GPUIDX);
  TestSoftmaxMultiClassObjGPair(&ctx);
}

TEST(Objective, DeclareUnifiedTest(SoftmaxMultiClassBasic)) {
  auto ctx = MakeCUDACtx(GPUIDX);
  TestSoftmaxMultiClassBasic(&ctx);
}

TEST(Objective, DeclareUnifiedTest(SoftprobMultiClassBasic)) {
  Context ctx = MakeCUDACtx(GPUIDX);
  TestSoftprobMultiClassBasic(&ctx);
}
}  // namespace xgboost
