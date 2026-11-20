/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <limits>  // for numeric_limits

#include "../../../src/tree/param.h"       // for TrainParam
#include "../../../src/tree/split_gain.h"  // for CalcGain, CalcWeight, NodeLossChange

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

TEST(SplitGain, NodeLossChange) {
  TrainParam param;
  param.UpdateAllowUnknown(xgboost::Args{{"reg_lambda", "1"}, {"min_child_weight", "1"}});
  constexpr double kNegInf = -std::numeric_limits<double>::infinity();
  GradientPairPrecise const parent{3.0, 4.0}, left{1.0, 2.0}, right{2.0, 2.0};
  auto gain = [&](GradientPairPrecise const &sum) {
    return CalcGain(param, sum.GetGrad(), sum.GetHess());
  };
  auto gain_at = [&](GradientPairPrecise const &sum, float w) {
    return CalcGainGivenWeight(param, sum.GetGrad(), sum.GetHess(), w);
  };

  // Without constraints, the closed-form gains.
  NodeLossChange const free{param, parent};
  EXPECT_EQ(free.LossChange(left, right, 0), gain(left) + gain(right) - gain(parent));
  EXPECT_EQ(free.TargetLossChange(left, right, 0), gain(left) + gain(right) - gain(parent));
  EXPECT_EQ(free.LossChange(left, GradientPairPrecise{2.0, 0.5}, 0), kNegInf);

  // With constraints but unbounded, the gains at the weights -1/3 and -2/3.
  constexpr float kMax = std::numeric_limits<float>::max();
  NodeLossChange const unbounded{param, parent, -kMax, kMax};
  float w_left = unbounded.Weight(left), w_right = unbounded.Weight(right),
        w_parent = unbounded.Weight(parent);
  EXPECT_FLOAT_EQ(w_left, -1.0f / 3.0f);
  EXPECT_FLOAT_EQ(w_right, -2.0f / 3.0f);
  auto ordered = gain_at(left, w_left) + gain_at(right, w_right) - gain_at(parent, w_parent);
  EXPECT_EQ(unbounded.LossChange(left, right, -1), ordered);
  EXPECT_EQ(unbounded.LossChange(left, right, 0), ordered);
  EXPECT_EQ(unbounded.TargetLossChange(left, right, -1), ordered);
  // Out of order: rejected for a single target, the pooled weight -3 / (4 + 2) for multiple.
  EXPECT_EQ(unbounded.LossChange(left, right, 1), kNegInf);
  auto pooled = gain_at(left, -0.5f) + gain_at(right, -0.5f) - gain_at(parent, w_parent);
  EXPECT_EQ(unbounded.TargetLossChange(left, right, 1), pooled);

  // The bounds of the node clamp the weights of the children and the node.
  NodeLossChange const bounded{param, parent, -0.4f, 0.0f};
  EXPECT_EQ(bounded.Weight(right), -0.4f);
  EXPECT_EQ(bounded.Weight(parent), -0.4f);
  EXPECT_EQ(bounded.LossChange(left, right, -1),
            gain_at(left, w_left) + gain_at(right, -0.4f) - gain_at(parent, -0.4f));

  // Equal weights can't improve on the node: the loss change is -lambda w^2.
  NodeLossChange const tight{param, parent, -0.2f, 0.0f};
  EXPECT_EQ(tight.LossChange(left, right, 0), kNegInf);
  double w = -0.2f;
  EXPECT_NEAR(tight.TargetLossChange(left, right, 0), -w * w, 1e-8);
}
}  // namespace xgboost::tree
