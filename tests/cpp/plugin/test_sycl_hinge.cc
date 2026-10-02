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
  for (auto device : {"sycl", "sycl:cpu"}) {
    Context ctx;
    ctx.UpdateAllowUnknown(Args{{"device", device}});
    TestHingeObj(&ctx);
  }
}

// The hinge kernels are registered by the SYCL plugin. Without those registrations
// DispatchKernel silently falls back to the CPU variant, so the numeric test above passes
// either way. Assert the SYCL variants exist so a lost registration fails loudly.
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
