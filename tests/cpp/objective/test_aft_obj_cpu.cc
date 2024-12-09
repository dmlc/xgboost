/**
 * Copyright 2020-2024, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <limits>
#include <cmath>

#include "xgboost/objective.h"
#include "xgboost/logging.h"
#include "../helpers.h"
#include "test_aft_obj.h"

namespace xgboost::common {
TEST(Objective, DeclareUnifiedTest(AFTObjConfiguration)) {
  auto ctx = MakeCUDACtx(GPUIDX);
  TestAFTObjConfiguration(&ctx);
}

TEST(Objective, DeclareUnifiedTest(AFTObjGPairUncensoredLabels)) {
  auto ctx = MakeCUDACtx(GPUIDX);
  TestAFTObjGPairUncensoredLabels(&ctx);
}

TEST(Objective, DeclareUnifiedTest(AFTObjGPairLeftCensoredLabels)) {
  auto ctx = MakeCUDACtx(GPUIDX);
  TestAFTObjGPairLeftCensoredLabels(&ctx);
}

TEST(Objective, DeclareUnifiedTest(AFTObjGPairRightCensoredLabels)) {
  auto ctx = MakeCUDACtx(GPUIDX);
  TestAFTObjGPairRightCensoredLabels(&ctx);
}

TEST(Objective, DeclareUnifiedTest(AFTObjGPairIntervalCensoredLabels)) {
  auto ctx = MakeCUDACtx(GPUIDX);
  TestAFTObjGPairIntervalCensoredLabels(&ctx);
}

}  // namespace xgboost::common
