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

TEST(TreeEvaluator, PropagateVectorBounds) {
  for (auto monotone_direction : {1, -1}) {
    SCOPED_TRACE(monotone_direction);
    TrainParam param;
    auto constraint = monotone_direction > 0 ? "(1, 0)" : "(-1, 0)";
    param.Init(Args{{"min_child_weight", "0"},
                    {"reg_alpha", "0"},
                    {"reg_lambda", "0"},
                    {"monotone_constraints", constraint}});

    // Unconstrained updates for the four (f0, f1) groups. Negating them tests decreasing f0.
    constexpr float kUpdate[4][2]{
        {-4.0f, 2.0f},  // f0=0, f1=0
        {2.0f, 2.0f},   // f0=0, f1=1
        {4.0f, 1.0f},   // f0=1, f1=0
        {-2.0f, 1.0f},  // f0=1, f1=1
    };
    linalg::Matrix<GradientPairPrecise> leaf_stats({4, 2}, DeviceOrd::CPU());
    auto h_leaf_stats = leaf_stats.HostView();
    for (std::size_t row = 0; row < 4; ++row) {
      for (bst_target_t target = 0; target < 2; ++target) {
        h_leaf_stats(row, target) = {-monotone_direction * kUpdate[row][target], 1.0};
      }
    }

    // The two f0 groups have mean updates (-1, 2) and (1, 1). The second target
    // crosses, so its two weights are both projected to 1.5.
    linalg::Matrix<GradientPairPrecise> root_stats({2, 2}, DeviceOrd::CPU());
    auto h_root_stats = root_stats.HostView();
    for (bst_target_t target = 0; target < 2; ++target) {
      h_root_stats(0, target) = h_leaf_stats(0, target) + h_leaf_stats(1, target);
      h_root_stats(1, target) = h_leaf_stats(2, target) + h_leaf_stats(3, target);
    }

    TreeEvaluator tree_evaluator{param, 2, DeviceOrd::CPU(), 2};
    auto evaluator = tree_evaluator.GetEvaluator();
    linalg::Matrix<float> root_weight({2, 2}, DeviceOrd::CPU());
    evaluator.CalcSplitWeights(
        param, 0, 0, root_stats.Slice(0, linalg::All()), root_stats.Slice(1, linalg::All()),
        root_weight.Slice(0, linalg::All()), root_weight.Slice(1, linalg::All()));
    auto h_root_weight = root_weight.HostView();
    EXPECT_FLOAT_EQ(h_root_weight(0, 0), monotone_direction * -1.0f);
    EXPECT_FLOAT_EQ(h_root_weight(0, 1), monotone_direction * 1.5f);
    EXPECT_FLOAT_EQ(h_root_weight(1, 0), monotone_direction * 1.0f);
    EXPECT_FLOAT_EQ(h_root_weight(1, 1), monotone_direction * 1.5f);

    tree_evaluator.AddSplit(0, 1, 2, 0, root_weight.Slice(0, linalg::All()),
                            root_weight.Slice(1, linalg::All()));
    evaluator = tree_evaluator.GetEvaluator();

    // f1 has no constraint, but each split inherits the target-specific bounds from f0.
    linalg::Matrix<float> leaf_weight({4, 2}, DeviceOrd::CPU());
    evaluator.CalcSplitWeights(
        param, 1, 1, leaf_stats.Slice(0, linalg::All()), leaf_stats.Slice(1, linalg::All()),
        leaf_weight.Slice(0, linalg::All()), leaf_weight.Slice(1, linalg::All()));
    evaluator.CalcSplitWeights(
        param, 2, 1, leaf_stats.Slice(2, linalg::All()), leaf_stats.Slice(3, linalg::All()),
        leaf_weight.Slice(2, linalg::All()), leaf_weight.Slice(3, linalg::All()));

    constexpr float kExpected[4][2]{{-4.0f, 1.5f}, {0.0f, 1.5f}, {4.0f, 1.5f}, {0.0f, 1.5f}};
    auto h_leaf_weight = leaf_weight.HostView();
    for (std::size_t row = 0; row < 4; ++row) {
      for (bst_target_t target = 0; target < 2; ++target) {
        EXPECT_FLOAT_EQ(h_leaf_weight(row, target), monotone_direction * kExpected[row][target]);
      }
    }
  }
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
