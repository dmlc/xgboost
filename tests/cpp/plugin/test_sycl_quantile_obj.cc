/**
 * Copyright 2024-2026 by XGBoost contributors
 */
#include <gtest/gtest.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include <xgboost/objective.h>
#pragma GCC diagnostic pop
#include <xgboost/context.h>

#include "../../../src/common/kernel.h"           // for GetKernelRegistry
#include "../../../src/objective/quantile_obj.h"  // for QuantileGradientKernel
#include "../helpers.h"
#include "../objective/test_quantile_obj.h"

namespace xgboost {
TEST(SyclObjective, DeclareUnifiedTest(Quantile)) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestQuantile(&ctx);
}

TEST(SyclObjective, DeclareUnifiedTest(QuantileIntercept)) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestQuantileIntercept(&ctx);
}

TEST(SyclObjective, QuantileKernelRegistration) {
  for (auto device : {DeviceOrd::kSyclDefault}) {
    EXPECT_NE(common::GetKernelRegistry<obj::QuantileGradientKernel>().Find(device), nullptr)
        << device;
    EXPECT_NE(common::GetKernelRegistry<obj::QuantileTransformKernel>().Find(device), nullptr)
        << device;
  }
}
}  // namespace xgboost
