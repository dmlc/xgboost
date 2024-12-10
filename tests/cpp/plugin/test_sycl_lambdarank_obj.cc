/**
 * Copyright 2024 by XGBoost Contributors
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include "../objective/test_lambdarank_obj.h"
#pragma GCC diagnostic pop

#include <gtest/gtest.h>

#include "xgboost/context.h"

namespace xgboost::obj {
TEST(SyclObjective, LambdaRankNDCGJsonIO) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestNDCGJsonIO(&ctx);
}

TEST(SyclObjective, LambdaRankTestNDCGGPair) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestNDCGGPair(&ctx);
}

TEST(SyclObjective, LambdaRankUnbiasedNDCG) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestUnbiasedNDCG(&ctx);
}

TEST(SyclObjective, LambdaRankMAPStat) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestMAPStat(&ctx);
}

TEST(SyclObjective, LambdaRankMAPGPair) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestMAPGPair(&ctx);
}

}  // namespace xgboost::obj

