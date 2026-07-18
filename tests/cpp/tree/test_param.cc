/**
 * Copyright 2014-2026, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include "../../../src/tree/param.h"
namespace xgboost::tree {
namespace {
void Clear(std::vector<int> &vals, std::stringstream &ss) {  // NOLINT
  vals.clear();
  ss.flush();
  ss.clear();
  ss.str("");
}
}  // namespace

TEST(Param, VectorIOStream) {
  std::vector<int> vals = {3, 2, 1};
  std::stringstream ss;
  std::vector<int> vals_in;

  ss << vals;
  EXPECT_EQ(ss.str(), "(3,2,1)");

  ss >> vals_in;
  EXPECT_EQ(vals_in, vals);

  Clear(vals, ss);
  vals = {1};
  ss << vals;
  EXPECT_EQ(ss.str(), "(1,)");
}

TEST(Param, VectorStreamRead) {
  std::vector<int> vals = {3, 2, 1};
  std::stringstream ss;
  std::vector<int> vals_in;

  Clear(vals_in, ss);
  ss << "(3, 2, 1)";
  ss >> vals_in;
  EXPECT_EQ(vals_in, vals);

  Clear(vals_in, ss);
  ss << "(3L,2L,1L)";
  ss >> vals_in;
  EXPECT_EQ(vals_in, vals);

  Clear(vals_in, ss);
  ss << " (3,2,1,)";
  ss >> vals_in;
  EXPECT_EQ(vals_in, vals);

  Clear(vals_in, ss);
  ss << " ( 3, 2,1 )";
  ss >> vals_in;
  EXPECT_EQ(vals_in, vals);

  Clear(vals_in, ss);
  ss << " ( 3, 2,1 ) ";
  ss >> vals_in;
  EXPECT_EQ(vals_in, vals);

  Clear(vals_in, ss);
  ss << " 321 ";
  ss >> vals_in;
  EXPECT_EQ(vals_in[0], 321);

  Clear(vals_in, ss);
  ss << "(3.0,2,1)";
  ss >> vals_in;
  EXPECT_NE(vals_in, vals);

  Clear(vals_in, ss);
  ss << "1a";
  ss >> vals_in;
  EXPECT_NE(vals_in, vals);

  Clear(vals_in, ss);
  ss << "abcde";
  ss >> vals_in;
  EXPECT_NE(vals_in, vals);

  Clear(vals_in, ss);
  ss << "(3,2,1";
  ss >> vals_in;
  EXPECT_NE(vals_in, vals);

  Clear(vals_in, ss);
  vals_in.emplace_back(3);
  ss << "( )";
  ss >> vals_in;
  ASSERT_TRUE(ss.good());
}

TEST(Param, SplitEntry) {
  SplitEntry se1;
  EXPECT_FALSE(se1.NeedReplace(-1, 100));

  SplitEntry se2;
  EXPECT_FALSE(se1.Update(se2));
  EXPECT_FALSE(se2.Update(-1, 100, 0, true, false, GradStats(), GradStats()));
  ASSERT_TRUE(se2.Update(1, 100, 0, true, false, GradStats(), GradStats()));
  ASSERT_TRUE(se1.Update(se2));

  SplitEntry se3;
  se3.Update(2, 101, 0, false, false, GradStats(), GradStats());
  se2.Update(se3);
  EXPECT_EQ(se2.SplitIndex(), 101);
  EXPECT_FALSE(se2.DefaultLeft());

  EXPECT_TRUE(se1.NeedReplace(3, 1));
}

TEST(Param, CalcGainWithL1AndMaxDeltaStep) {
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

TEST(Param, CalcVectorGainWithMaxDeltaStep) {
  TrainParam param;
  param.UpdateAllowUnknown(
      xgboost::Args{{"reg_alpha", "1"}, {"reg_lambda", "0"}, {"max_delta_step", "0.5"}});

  xgboost::linalg::Vector<xgboost::GradientPairPrecise> stats({2}, xgboost::DeviceOrd::CPU());
  auto h_stats = stats.HostView();
  h_stats(0) = {8.0, 2.0};
  h_stats(1) = {-2.0, 4.0};

  xgboost::linalg::Vector<float> weight({2}, xgboost::DeviceOrd::CPU());
  auto h_weight = weight.HostView();
  CalcWeight(param, h_stats, h_weight);
  ASSERT_FLOAT_EQ(h_weight(0), -0.5f);
  ASSERT_FLOAT_EQ(h_weight(1), 0.25f);

  // 6.5 from the clipped first target and 0.25 from the second target.
  EXPECT_DOUBLE_EQ(CalcGainGivenWeight(param, h_stats, h_weight), 6.75);
}
}  // namespace xgboost::tree
