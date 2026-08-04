/**
 * Copyright 2018-2023, XGBoost Contributors
 */
#include <xgboost/context.h>
#include <xgboost/objective.h>

#include <limits>

#include "../../../src/common/linalg_op.h"
#include "../helpers.h"
#include "test_hinge.h"

namespace xgboost {

TEST(Objective, DeclareUnifiedTest(HingeObj)) {
  Context ctx = MakeCUDACtx(GPUIDX);
  TestHingeObj(&ctx);
}

#if !defined(__CUDACC__)
TEST(Objective, HingeKernelCPUFallback) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", DeviceSym::SyclDefault()}});
  TestHingeObj(&ctx);
}
#endif  // !defined(__CUDACC__)
}  // namespace xgboost
