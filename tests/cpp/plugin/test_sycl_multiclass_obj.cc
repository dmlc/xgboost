/*!
 * Copyright 2018-2026 XGBoost contributors
 */
#include <gtest/gtest.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include <xgboost/context.h>
#pragma GCC diagnostic pop

#include "../../../src/common/kernel.h"             // for GetKernelRegistry
#include "../../../src/objective/multiclass_obj.h"  // for MulticlassGradientKernel
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
  TestSoftmaxMultiClassBasic(&ctx);
}

TEST(SyclObjective, SoftprobMultiClassBasic) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestSoftprobMultiClassBasic(&ctx);
}

TEST(SyclObjective, MulticlassKernelRegistration) {
  for (auto device : {DeviceOrd::kSyclDefault}) {
    EXPECT_NE(common::GetKernelRegistry<obj::MulticlassGradientKernel>().Find(device), nullptr)
        << device;
    EXPECT_NE(common::GetKernelRegistry<obj::MulticlassInitEstimationKernel>().Find(device),
              nullptr)
        << device;
    EXPECT_NE(common::GetKernelRegistry<obj::MulticlassTransformKernel>().Find(device), nullptr)
        << device;
    EXPECT_NE(common::GetKernelRegistry<obj::MulticlassValidationKernel>().Find(device), nullptr)
        << device;
    EXPECT_NE(common::GetKernelRegistry<obj::MulticlassCenterKernel>().Find(device), nullptr)
        << device;
  }
}
}  // namespace xgboost
