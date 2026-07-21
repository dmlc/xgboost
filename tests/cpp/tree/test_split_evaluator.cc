/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include "../../../src/tree/param.h"  // for TrainParam
#include "../../../src/tree/split_evaluator.h"
#include "xgboost/linalg.h"  // for Vector

namespace xgboost::tree {
TEST(TreeEvaluator, CalcVectorGainWithMaxDeltaStep) {
  TrainParam param;
  param.UpdateAllowUnknown(
      xgboost::Args{{"reg_alpha", "1"}, {"reg_lambda", "0"}, {"max_delta_step", "0.5"}});

  linalg::Vector<xgboost::GradientPairPrecise> stats({2}, xgboost::DeviceOrd::CPU());
  auto h_stats = stats.HostView();
  h_stats(0) = {8.0, 2.0};
  h_stats(1) = {-2.0, 4.0};

  linalg::Vector<float> weight({2}, xgboost::DeviceOrd::CPU());
  auto h_weight = weight.HostView();
  TreeEvaluator tree_evaluator{param, 1, DeviceOrd::CPU()};
  auto evaluator = tree_evaluator.GetEvaluator();
  evaluator.CalcWeight(0, param, h_stats, h_weight);
  ASSERT_FLOAT_EQ(h_weight(0), -0.5f);
  ASSERT_FLOAT_EQ(h_weight(1), 0.25f);

  // 6.5 from the clipped first target and 0.25 from the second target.
  EXPECT_DOUBLE_EQ(evaluator.CalcGainGivenWeight(param, h_stats, h_weight), 6.75);
  EXPECT_DOUBLE_EQ(evaluator.CalcGain(0, param, h_stats), 6.75);

  param.monotone_constraints = {0};
  TreeEvaluator all_zero{param, 1, DeviceOrd::CPU()};
  EXPECT_DOUBLE_EQ(all_zero.GetEvaluator().CalcGain(0, param, h_stats), 6.75);
}

TEST(TreeEvaluator, CalcMonotoneVectorSplitWeights) {
  TrainParam param;
  param.Init(Args{{"min_child_weight", "0"},
                  {"reg_alpha", "0"},
                  {"reg_lambda", "0"},
                  {"monotone_constraints", "(1)"}});

  linalg::Vector<GradientPairPrecise> left({2}, DeviceOrd::CPU());
  linalg::Vector<GradientPairPrecise> right({2}, DeviceOrd::CPU());
  auto h_left = left.HostView();
  auto h_right = right.HostView();
  // Crossing for an increasing constraint: 2 > 1.
  h_left(0) = {-2.0, 1.0};
  h_right(0) = {-1.0, 1.0};
  // Already ordered: -1 < 1.
  h_left(1) = {1.0, 1.0};
  h_right(1) = {-1.0, 1.0};

  TreeEvaluator tree_evaluator{param, 1, DeviceOrd::CPU(), 2};
  auto evaluator = tree_evaluator.GetEvaluator();
  linalg::Vector<float> left_weight({2}, DeviceOrd::CPU());
  linalg::Vector<float> right_weight({2}, DeviceOrd::CPU());
  evaluator.CalcSplitWeights(param, 0, 0, h_left, h_right, left_weight.HostView(),
                             right_weight.HostView());

  auto h_left_weight = left_weight.HostView();
  auto h_right_weight = right_weight.HostView();
  EXPECT_FLOAT_EQ(h_left_weight(0), 1.5f);
  EXPECT_FLOAT_EQ(h_right_weight(0), 1.5f);
  EXPECT_FLOAT_EQ(h_left_weight(1), -1.0f);
  EXPECT_FLOAT_EQ(h_right_weight(1), 1.0f);
  EXPECT_DOUBLE_EQ(evaluator.CalcSplitGain(param, 0, 0, h_left, h_right), 6.5);

  // A crossing pair retains two copies of each regularizer.
  param.reg_alpha = 0.5f;
  param.reg_lambda = 1.0f;
  auto [l_w, r_w] = evaluator.CalcSplitWeights(param, 0, 0, 0, h_left(0), h_right(0));
  EXPECT_FLOAT_EQ(l_w, 0.5f);
  EXPECT_FLOAT_EQ(r_w, 0.5f);
  // max_delta_step applies once to the common weight.
  param.max_delta_step = 0.4f;
  std::tie(l_w, r_w) = evaluator.CalcSplitWeights(param, 0, 0, 0, h_left(0), h_right(0));
  EXPECT_FLOAT_EQ(l_w, 0.4f);
  EXPECT_FLOAT_EQ(r_w, 0.4f);
}

TEST(TreeEvaluator, PropagateVectorBounds) {
  TrainParam param;
  param.Init(Args{{"reg_lambda", "0"}, {"monotone_constraints", "(1, 0)"}});

  TreeEvaluator tree_evaluator{param, 2, DeviceOrd::CPU(), 2};
  linalg::Vector<float> left_weight({2}, DeviceOrd::CPU());
  linalg::Vector<float> right_weight({2}, DeviceOrd::CPU());
  left_weight.HostView()(0) = 2.0f;
  left_weight.HostView()(1) = -4.0f;
  right_weight.HostView()(0) = 4.0f;
  right_weight.HostView()(1) = -2.0f;
  tree_evaluator.AddSplit(0, 1, 2, 0, left_weight.HostView(), right_weight.HostView());

  // An unconstrained descendant split must inherit the target-specific intervals.
  tree_evaluator.AddSplit(1, 3, 4, 1, left_weight.HostView(), left_weight.HostView());
  auto evaluator = tree_evaluator.GetEvaluator();

  EXPECT_FLOAT_EQ(evaluator.CalcWeight(4, 0, param, GradientPairPrecise{-10.0, 1.0}), 3.0f);
  EXPECT_FLOAT_EQ(evaluator.CalcWeight(4, 1, param, GradientPairPrecise{-1.0, 1.0}), -3.0f);
  EXPECT_FLOAT_EQ(evaluator.CalcWeight(2, 0, param, GradientPairPrecise{1.0, 1.0}), 3.0f);
  EXPECT_FLOAT_EQ(evaluator.CalcWeight(2, 1, param, GradientPairPrecise{10.0, 1.0}), -3.0f);
}

TEST(TreeEvaluator, ConstrainedVectorGainWithZeroHessianTarget) {
  TrainParam param;
  param.Init(Args{{"min_child_weight", "0"},
                  {"reg_alpha", "0"},
                  {"reg_lambda", "1"},
                  {"monotone_constraints", "(-1)"}});

  linalg::Vector<GradientPairPrecise> left({2}, DeviceOrd::CPU());
  linalg::Vector<GradientPairPrecise> right({2}, DeviceOrd::CPU());
  auto h_left = left.HostView();
  auto h_right = right.HostView();
  h_left(0) = {0.0, 0.0};
  h_right(0) = {-2.0, 1.0};
  // Keep the normalized child Hessians valid.
  h_left(1) = {0.0, 2.0};
  h_right(1) = {0.0, 1.0};

  TreeEvaluator tree_evaluator{param, 1, DeviceOrd::CPU(), 2};
  auto evaluator = tree_evaluator.GetEvaluator();
  auto [l_w, r_w] = evaluator.CalcSplitWeights(param, 0, 0, 0, h_left(0), h_right(0));
  EXPECT_NEAR(l_w, 2.0f / 3.0f, kRtEps);
  EXPECT_NEAR(r_w, 2.0f / 3.0f, kRtEps);

  // Raw explicit gains include both regularized leaves: -4/9 + 16/9 = 4/3.
  EXPECT_NEAR(evaluator.CalcSplitGain(param, 0, 0, h_left, h_right), 4.0 / 3.0, kRtEps);
}
}  // namespace xgboost::tree
