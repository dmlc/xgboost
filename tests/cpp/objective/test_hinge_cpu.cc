/**
 * Copyright 2018-2023, XGBoost Contributors
 */
#include <xgboost/objective.h>
#include <xgboost/context.h>
#include <limits>

#include "../helpers.h"
#include "test_hinge.h"
#include "../../../src/common/linalg_op.h"

namespace xgboost {

TEST(Objective, DeclareUnifiedTest(HingeObj)) {
  Context ctx = MakeCUDACtx(GPUIDX);
  TestHingeObj(&ctx);
}
}  // namespace xgboost
