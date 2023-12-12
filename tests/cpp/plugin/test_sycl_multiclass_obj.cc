/*!
 * Copyright 2018-2023 XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>

#include "../objective/test_multiclass_obj.h"

namespace xgboost {

TEST(SyclObjective, SoftmaxMultiClassObjGPair) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestSoftmaxMultiClassObjGPair(&ctx);
}

TEST(SyclObjective, SoftmaxMultiClassBasic) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestSoftmaxMultiClassObjGPair(&ctx);
}

TEST(SyclObjective, SoftprobMultiClassBasic) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestSoftprobMultiClassBasic(&ctx);
}
}  // namespace xgboost
