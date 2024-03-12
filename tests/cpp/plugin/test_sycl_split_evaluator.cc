/**
 * Copyright 2020-2024 by XGBoost contributors
 */
#include <gtest/gtest.h>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include "../../../plugin/sycl/tree/split_evaluator.h"
#pragma GCC diagnostic pop

#include "../../../plugin/sycl/device_manager.h"
#include "../helpers.h"

namespace xgboost::sycl::tree {

template<typename GradientSumT>
void BasicTestSplitEvaluator(const std::string& monotone_constraints, bool has_constrains) {
  const size_t n_columns = 2;

  xgboost::tree::TrainParam param;
  param.UpdateAllowUnknown(Args{{"min_child_weight", "0"},
                                {"reg_lambda", "0"},
                                {"monotone_constraints", monotone_constraints}});

  DeviceManager device_manager;
  auto qu = device_manager.GetQueue(DeviceOrd::SyclDefault());

  TreeEvaluator<GradientSumT> tree_evaluator(qu, param, n_columns);
  {
    // Check correctness of has_constrains flag
    ASSERT_EQ(tree_evaluator.HasConstraint(), has_constrains);
  }

  auto split_evaluator = tree_evaluator.GetEvaluator();
  {
    // Check if params were inititialised correctly
    ASSERT_EQ(split_evaluator.param.min_child_weight, param.min_child_weight);
    ASSERT_EQ(split_evaluator.param.reg_lambda, param.reg_lambda);
    ASSERT_EQ(split_evaluator.param.reg_alpha, param.reg_alpha);
    ASSERT_EQ(split_evaluator.param.max_delta_step, param.max_delta_step);
  }
}

template<typename GradientSumT>
void TestSplitEvaluator(const std::string& monotone_constraints) {
  const size_t n_columns = 2;

  xgboost::tree::TrainParam param;
  param.UpdateAllowUnknown(Args{{"min_child_weight", "0"},
                                {"reg_lambda", "0"},
                                {"monotone_constraints", monotone_constraints}});

  DeviceManager device_manager;
  auto qu = device_manager.GetQueue(DeviceOrd::SyclDefault());

  TreeEvaluator<GradientSumT> tree_evaluator(qu, param, n_columns);
  auto split_evaluator = tree_evaluator.GetEvaluator();
  {
    // Test ThresholdL1
    const GradientSumT alpha = 0.5;
    {
      const GradientSumT val = 0.0;
      const auto trh = split_evaluator.ThresholdL1(val, alpha);
      ASSERT_EQ(trh, 0.0);
    }

    {
      const GradientSumT val = 1.0;
      const auto trh = split_evaluator.ThresholdL1(val, alpha);
      ASSERT_EQ(trh, val - alpha);
    }

    {
      const GradientSumT val = -1.0;
      const auto trh = split_evaluator.ThresholdL1(val, alpha);
      ASSERT_EQ(trh, val + alpha);
    }
  }

  {
    constexpr float eps = 1e-8;
    tree_evaluator.AddSplit(0, 1, 2, 0, 0.3, 0.7);

    GradStats<GradientSumT> left(0.1, 0.2);
    GradStats<GradientSumT> right(0.3, 0.4);
    bst_node_t nidx = 0;
    bst_feature_t fidx = 0;

    GradientSumT wleft  = split_evaluator.CalcWeight(nidx, left);
    // wleft = -grad/hess = -0.1/0.2
    EXPECT_NEAR(wleft, -0.5, eps);
    GradientSumT wright = split_evaluator.CalcWeight(nidx, right);
    // wright = -grad/hess = -0.3/0.4
    EXPECT_NEAR(wright, -0.75, eps);

    GradientSumT gweight_left = split_evaluator.CalcGainGivenWeight(nidx, left, wleft);
    // gweight_left = left.grad**2 / left.hess = 0.1*0.1/0.2 = 0.05
    EXPECT_NEAR(gweight_left, 0.05, eps);
    // gweight_left = right.grad**2 / right.hess = 0.3*0.3/0.4 = 0.225
    GradientSumT gweight_right = split_evaluator.CalcGainGivenWeight(nidx, right, wright);
    EXPECT_NEAR(gweight_right, 0.225, eps);

    GradientSumT split_gain = split_evaluator.CalcSplitGain(nidx, fidx, left, right);
    if (!tree_evaluator.HasConstraint()) {
      EXPECT_NEAR(split_gain, gweight_left + gweight_right, eps);
    } else {
      // Parameters are chosen to have -inf here
      ASSERT_EQ(split_gain, -std::numeric_limits<GradientSumT>::infinity());
    }
  }
}

TEST(SyclSplitEvaluator, BasicTest) {
  BasicTestSplitEvaluator<float>("( 0,  0)", false);
  BasicTestSplitEvaluator<float>("( 1,  0)", true);
  BasicTestSplitEvaluator<float>("( 0,  1)", true);
  BasicTestSplitEvaluator<float>("(-1,  0)", true);
  BasicTestSplitEvaluator<float>("( 0, -1)", true);
  BasicTestSplitEvaluator<float>("( 1,  1)", true);
  BasicTestSplitEvaluator<float>("(-1, -1)", true);
  BasicTestSplitEvaluator<float>("( 1, -1)", true);
  BasicTestSplitEvaluator<float>("(-1,  1)", true);
}

TEST(SyclSplitEvaluator, TestMath) {
  // Without constraints
  TestSplitEvaluator<float>("( 0,  0)");
  // With constraints
  TestSplitEvaluator<float>("( 1,  0)");
}

}  // namespace xgboost::sycl::tree
