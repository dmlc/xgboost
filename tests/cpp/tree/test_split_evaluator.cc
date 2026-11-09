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
  TreeEvaluator tree_evaluator{param, 1, DeviceOrd::CPU(), 1u};
  auto evaluator = tree_evaluator.GetEvaluator();
  evaluator.CalcWeight(0, param, h_stats, h_weight);
  ASSERT_FLOAT_EQ(h_weight(0), -0.5f);
  ASSERT_FLOAT_EQ(h_weight(1), 0.25f);

  // 6.5 from the clipped first target and 0.25 from the second target.
  EXPECT_DOUBLE_EQ(evaluator.CalcGainGivenWeight(param, h_stats, h_weight), 6.75);
  EXPECT_DOUBLE_EQ(evaluator.CalcGain(0, param, h_stats), 6.75);

  param.monotone_constraints = {0};
  TreeEvaluator all_zero{param, 1, DeviceOrd::CPU(), 1u};
  EXPECT_DOUBLE_EQ(all_zero.GetEvaluator().CalcGain(0, param, h_stats), 6.75);
}

TEST(TreeEvaluator, PooledWeightUsesTwoLeafRegularization) {
  TrainParam param;
  param.Init(Args{{"min_child_weight", "0"},
                  {"reg_alpha", "0.25"},
                  {"reg_lambda", "1"},
                  {"monotone_constraints", "(1)"}});

  linalg::Vector<GradientPairPrecise> left({2}, DeviceOrd::CPU());
  linalg::Vector<GradientPairPrecise> right({2}, DeviceOrd::CPU());
  auto h_left = left.HostView();
  auto h_right = right.HostView();
  h_left(0) = {-2.0, 1.0};
  h_right(0) = {-1.0, 1.0};
  // A second target with zero weight contributes no gain or regularization cost.
  h_left(1) = {0.0, 1.0};
  h_right(1) = {0.0, 1.0};

  TreeEvaluator tree_evaluator{param, 1, DeviceOrd::CPU(), 2};
  auto evaluator = tree_evaluator.GetEvaluator();
  auto pooled = evaluator.CalcPooledWeight(param, 0, 0, h_left(0), h_right(0));
  EXPECT_FLOAT_EQ(pooled, 0.625f);

  // Combining two equal-valued leaves is an ordinary one-leaf calculation with both
  // regularization penalties doubled.
  auto pooled_param = param;
  pooled_param.reg_alpha *= 2.0f;
  pooled_param.reg_lambda *= 2.0f;
  auto sum = h_left(0) + h_right(0);
  EXPECT_FLOAT_EQ(pooled, CalcWeight(pooled_param, sum));

  auto gain = evaluator.CalcGainGivenWeight(param, h_left(0), pooled) +
              evaluator.CalcGainGivenWeight(param, h_right(0), pooled);
  EXPECT_DOUBLE_EQ(gain, CalcGain(pooled_param, sum));
  EXPECT_DOUBLE_EQ(evaluator.CalcSplitGain(param, 0, 0, h_left, h_right), gain);
}

TEST(TreeEvaluator, PropagateVectorBounds) {
  for (auto direction : {1, -1}) {
    SCOPED_TRACE(direction);
    TrainParam param;
    param.Init(Args{{"min_child_weight", "0"},
                    {"reg_alpha", "0"},
                    {"reg_lambda", "0"},
                    {"monotone_constraints", direction > 0 ? "(1, 0)" : "(-1, 0)"}});

    // Rows are the (f0, f1) groups: (0, 0), (0, 1), (1, 0), (1, 1).
    // Negating the updates tests decreasing f0.
    constexpr float kUpdate[4][2]{{-4, 2}, {2, 2}, {4, 1}, {-2, 1}};
    GradientPairPrecise leaf_stats[4][2];
    GradientPairPrecise root_stats[2][2];
    for (bst_target_t t = 0; t < 2; ++t) {
      for (std::size_t row = 0; row < 4; ++row) {
        leaf_stats[row][t] = {-direction * kUpdate[row][t], 1.0};
      }
      root_stats[0][t] = leaf_stats[0][t] + leaf_stats[1][t];
      root_stats[1][t] = leaf_stats[2][t] + leaf_stats[3][t];
    }

    TreeEvaluator tree_evaluator{param, 2, DeviceOrd::CPU(), 2};
    auto evaluator = tree_evaluator.GetEvaluator();
    auto left = linalg::MakeVec(root_stats[0], 2);
    auto right = linalg::MakeVec(root_stats[1], 2);
    float root_weight[2][2];
    auto left_weight = linalg::MakeVec(root_weight[0], 2);
    auto right_weight = linalg::MakeVec(root_weight[1], 2);
    evaluator.CalcSplitWeights(param, 0, 0, left, right, left_weight, right_weight);

    // Mean updates are (-1, 2) and (1, 1). Pool the crossing second target to 1.5.
    EXPECT_FLOAT_EQ(left_weight(0), direction * -1.0f);
    EXPECT_FLOAT_EQ(right_weight(0), direction * 1.0f);
    EXPECT_FLOAT_EQ(left_weight(1), direction * 1.5f);
    EXPECT_FLOAT_EQ(right_weight(1), direction * 1.5f);
    // Pooling must preserve the first target's gain: child score 4 + 9, parent score 0 + 9.
    EXPECT_DOUBLE_EQ(evaluator.CalcSplitGain(param, 0, 0, left, right), 13.0);
    tree_evaluator.AddSplit(0, 1, 2, 0, left_weight, right_weight);
    evaluator = tree_evaluator.GetEvaluator();

    // Both f1 splits inherit target-specific bounds from f0 despite being unconstrained.
    float leaf_weight[4][2];
    for (bst_node_t parent : {1, 2}) {
      auto row = 2 * (parent - 1);
      evaluator.CalcSplitWeights(param, parent, 1, linalg::MakeVec(leaf_stats[row], 2),
                                 linalg::MakeVec(leaf_stats[row + 1], 2),
                                 linalg::MakeVec(leaf_weight[row], 2),
                                 linalg::MakeVec(leaf_weight[row + 1], 2));
    }
    // Apply both splits together, with child IDs beyond the initial bounds storage.
    constexpr bst_node_t kFirstChild = 300;
    tree_evaluator.AddSplits(2, kFirstChild + 3, [&](std::size_t i, bst_target_t t) {
      auto parent = static_cast<bst_node_t>(i) + 1;
      auto child = kFirstChild + 2 * (parent - 1);
      return TreeEvaluator::SplitBounds{
          parent, child, child + 1, 1, leaf_weight[2 * i][t], leaf_weight[2 * i + 1][t]};
    });
    evaluator = tree_evaluator.GetEvaluator();

    // f0 bounds clamp target 0 at zero and target 1 at the pooled midpoint, 1.5.
    constexpr float kExpected[4][2]{{-4, 1.5f}, {0, 1.5f}, {4, 1.5f}, {0, 1.5f}};
    for (bst_node_t row = 0; row < 4; ++row) {
      for (bst_target_t t = 0; t < 2; ++t) {
        EXPECT_FLOAT_EQ(leaf_weight[row][t], direction * kExpected[row][t]);
        EXPECT_FLOAT_EQ(evaluator.CalcWeight(kFirstChild + row, t, param, leaf_stats[row][t]),
                        leaf_weight[row][t]);
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
