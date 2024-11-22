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
#include "../objective/test_hinge.h"

namespace xgboost {
TEST(SyclObjective, DeclareUnifiedTest(HingeObj)) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestHingeObj(&ctx);
}

}  // namespace xgboost
