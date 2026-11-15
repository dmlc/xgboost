/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include "../../../src/tree/param.h"       // for TrainParam
#include "../../../src/tree/split_gain.h"  // for CalcGain, CalcWeight, CalcPooledWeight

namespace xgboost::tree {
TEST(SplitGain, CalcGainWithL1AndMaxDeltaStep) {
  TrainParam param;
  param.UpdateAllowUnknown(xgboost::Args{{"reg_alpha", "1"}, {"reg_lambda", "1"}});

  constexpr float kGrad = 4.0f;
  constexpr float kHess = 1.0f;

  // Gain at the delta-clipped weight includes the complete L1 penalty.
  EXPECT_FLOAT_EQ(CalcGainGivenWeight(param, kGrad, kHess, -0.5f), 2.5f);

  // A non-binding max_delta_step must not change the unconstrained weight or gain.
  param.max_delta_step = 0.0f;
  EXPECT_FLOAT_EQ(CalcWeight(param, kGrad, kHess), -1.5f);
  EXPECT_FLOAT_EQ(CalcGain(param, kGrad, kHess), 4.5f);

  param.max_delta_step = 10.0f;
  EXPECT_FLOAT_EQ(CalcWeight(param, kGrad, kHess), -1.5f);
  EXPECT_FLOAT_EQ(CalcGain(param, kGrad, kHess), 4.5f);

  param.max_delta_step = 0.5f;
  EXPECT_FLOAT_EQ(CalcWeight(param, kGrad, kHess), -0.5f);
  EXPECT_FLOAT_EQ(CalcGain(param, kGrad, kHess), 2.5f);
}

TEST(SplitGain, PooledWeight) {
  TrainParam param;
  param.UpdateAllowUnknown(xgboost::Args{{"reg_alpha", "0.5"}, {"reg_lambda", "1"}});

  // Two leaves sharing one weight pay two L1 and two L2 penalties: -(4 - 1) / (2 + 2).
  constexpr double kGrad = 4.0, kHess = 2.0;
  EXPECT_DOUBLE_EQ(CalcPooledWeight(param, kGrad, kHess), -0.75);

  param.max_delta_step = 0.5f;
  EXPECT_DOUBLE_EQ(CalcPooledWeight(param, kGrad, kHess), -0.5);
}

TEST(SplitGain, IsValidSplit) {
  TrainParam param;
  param.UpdateAllowUnknown(xgboost::Args{{"min_child_weight", "1"}});
  EXPECT_TRUE(IsValidSplit(param, 1.0, 2.0));
  EXPECT_FALSE(IsValidSplit(param, 0.5, 2.0));
  EXPECT_FALSE(IsValidSplit(param, 2.0, 0.5));

  param.min_child_weight = 0.0f;
  // A child needs some Hessian even without a minimum.
  EXPECT_FALSE(IsValidSplit(param, 0.0, 2.0));
  EXPECT_TRUE(IsValidSplit(param, 1e-6, 2.0));
}
}  // namespace xgboost::tree
