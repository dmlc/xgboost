/**
 * Copyright 2024 by XGBoost contributors
 */
#include <gtest/gtest.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include <xgboost/objective.h>
#pragma GCC diagnostic pop
#include <xgboost/context.h>

#include "../helpers.h"
#include "../objective/test_aft_obj.h"

namespace xgboost::common {
TEST(SyclObjective, DeclareUnifiedTest(AFTObjConfiguration)) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestAFTObjConfiguration(&ctx);
}

TEST(SyclObjective, DeclareUnifiedTest(AFTObjGPairUncensoredLabels)) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestAFTObjGPairUncensoredLabels(&ctx);
}

TEST(SyclObjective, DeclareUnifiedTest(AFTObjGPairLeftCensoredLabels)) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestAFTObjGPairLeftCensoredLabels(&ctx);
}

TEST(SyclObjective, DeclareUnifiedTest(AFTObjGPairRightCensoredLabels)) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestAFTObjGPairRightCensoredLabels(&ctx);
}

TEST(SyclObjective, DeclareUnifiedTest(AFTObjGPairIntervalCensoredLabels)) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestAFTObjGPairIntervalCensoredLabels(&ctx);
}

}  // namespace xgboost::common
