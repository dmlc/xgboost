/*!
 * Copyright 2018-2023 XGBoost contributors
 */
#include <gtest/gtest.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include <xgboost/context.h>
#pragma GCC diagnostic pop

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
