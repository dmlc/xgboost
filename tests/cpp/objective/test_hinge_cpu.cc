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
}  // namespace xgboost
