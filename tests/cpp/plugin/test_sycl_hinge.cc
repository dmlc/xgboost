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

#include "../../../src/common/kernel.h"    // for GetKernelRegistry
#include "../../../src/objective/hinge.h"  // for HingeGradientKernel
#include "../helpers.h"
#include "../objective/test_hinge.h"

namespace xgboost {
TEST(SyclObjective, DeclareUnifiedTest(HingeObj)) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestHingeObj(&ctx);
}

TEST(SyclObjective, HingeKernelRegistration) {
  for (auto device : {DeviceOrd::kSyclDefault, DeviceOrd::kSyclCPU, DeviceOrd::kSyclGPU}) {
    EXPECT_NE(common::GetKernelRegistry<obj::HingeGradientKernel>().Find(device), nullptr)
        << device;
    EXPECT_NE(common::GetKernelRegistry<obj::HingePredTransformKernel>().Find(device), nullptr)
        << device;
    EXPECT_NE(common::GetKernelRegistry<obj::HingeValidationKernel>().Find(device), nullptr)
        << device;
  }
}
}  // namespace xgboost
