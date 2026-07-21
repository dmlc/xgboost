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
}
}  // namespace xgboost::tree
