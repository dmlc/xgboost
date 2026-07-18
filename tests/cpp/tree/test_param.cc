// Copyright by Contributors
#include <gtest/gtest.h>

#include "../../../src/tree/param.h"
#include "../helpers.h"

TEST(Param, VectorIOStream) {
  std::vector<int> vals = {3, 2, 1};
  std::stringstream ss;
  std::vector<int> vals_in;

  ss << vals;
  EXPECT_EQ(ss.str(), "(3,2,1)");

  ss >> vals_in;
  EXPECT_EQ(vals_in, vals);

  vals.clear();
  ss.flush();
  ss.clear();
  ss.str("");
  vals = {1};
  ss << vals;
  EXPECT_EQ(ss.str(), "(1,)");
}

TEST(Param, VectorStreamRead) {
  std::vector<int> vals = {3, 2, 1};
  std::stringstream ss;
  std::vector<int> vals_in;

  vals_in.clear();
  ss.flush();
  ss.clear();
  ss.str("");
  ss << "(3, 2, 1)";
  ss >> vals_in;
  EXPECT_EQ(vals_in, vals);

  vals_in.clear();
  ss.flush();
  ss.clear();
  ss.str("");
  ss << "(3L,2L,1L)";
  ss >> vals_in;
  EXPECT_EQ(vals_in, vals);

  vals_in.clear();
  ss.flush();
  ss.clear();
  ss.str("");
  ss << " (3,2,1,)";
  ss >> vals_in;
  EXPECT_EQ(vals_in, vals);

  vals_in.clear();
  ss.flush();
  ss.clear();
  ss.str("");
  ss << " ( 3, 2,1 )";
  ss >> vals_in;
  EXPECT_EQ(vals_in, vals);

  vals_in.clear();
  ss.flush();
  ss.clear();
  ss.str("");
  ss << " ( 3, 2,1 ) ";
  ss >> vals_in;
  EXPECT_EQ(vals_in, vals);

  vals_in.clear();
  ss.flush();
  ss.clear();
  ss.str("");
  ss << " 321 ";
  ss >> vals_in;
  EXPECT_EQ(vals_in[0], 321);

  vals_in.clear();
  ss.flush();
  ss.clear();
  ss.str("");
  ss << "(3.0,2,1)";
  ss >> vals_in;
  EXPECT_NE(vals_in, vals);

  vals_in.clear();
  ss.flush();
  ss.clear();
  ss.str("");
  ss << "1a";
  ss >> vals_in;
  EXPECT_NE(vals_in, vals);

  vals_in.clear();
  ss.flush();
  ss.clear();
  ss.str("");
  ss << "abcde";
  ss >> vals_in;
  EXPECT_NE(vals_in, vals);

  vals_in.clear();
  ss.flush();
  ss.clear();
  ss.str("");
  ss << "(3,2,1";
  ss >> vals_in;
  EXPECT_NE(vals_in, vals);

  vals_in.clear();
  ss.flush();
  ss.clear();
  ss.str("");
  vals_in.emplace_back(3);
  ss << "( )";
  ss >> vals_in;
  ASSERT_TRUE(ss.good());
}

TEST(Param, SplitEntry) {
  xgboost::tree::SplitEntry se1;
  EXPECT_FALSE(se1.NeedReplace(-1, 100));

  xgboost::tree::SplitEntry se2;
  EXPECT_FALSE(se1.Update(se2));
  EXPECT_FALSE(
      se2.Update(-1, 100, 0, true, false, xgboost::tree::GradStats(), xgboost::tree::GradStats()));
  ASSERT_TRUE(
      se2.Update(1, 100, 0, true, false, xgboost::tree::GradStats(), xgboost::tree::GradStats()));
  ASSERT_TRUE(se1.Update(se2));

  xgboost::tree::SplitEntry se3;
  se3.Update(2, 101, 0, false, false, xgboost::tree::GradStats(), xgboost::tree::GradStats());
  se2.Update(se3);
  EXPECT_EQ(se2.SplitIndex(), 101);
  EXPECT_FALSE(se2.DefaultLeft());

  EXPECT_TRUE(se1.NeedReplace(3, 1));
}

TEST(Param, CalcGainWithL1AndMaxDeltaStep) {
  xgboost::tree::TrainParam param;
  param.UpdateAllowUnknown(xgboost::Args{{"reg_alpha", "1"}, {"reg_lambda", "1"}});

  constexpr float kGrad = 4.0f;
  constexpr float kHess = 1.0f;

  // Gain at the delta-clipped weight includes the complete L1 penalty.
  EXPECT_FLOAT_EQ(xgboost::tree::CalcGainGivenWeight(param, kGrad, kHess, -0.5f), 2.5f);

  // A non-binding max_delta_step must not change the unconstrained weight or gain.
  param.max_delta_step = 0.0f;
  EXPECT_FLOAT_EQ(xgboost::tree::CalcWeight(param, kGrad, kHess), -1.5f);
  EXPECT_FLOAT_EQ(xgboost::tree::CalcGain(param, kGrad, kHess), 4.5f);

  param.max_delta_step = 10.0f;
  EXPECT_FLOAT_EQ(xgboost::tree::CalcWeight(param, kGrad, kHess), -1.5f);
  EXPECT_FLOAT_EQ(xgboost::tree::CalcGain(param, kGrad, kHess), 4.5f);

  param.max_delta_step = 0.5f;
  EXPECT_FLOAT_EQ(xgboost::tree::CalcWeight(param, kGrad, kHess), -0.5f);
  EXPECT_FLOAT_EQ(xgboost::tree::CalcGain(param, kGrad, kHess), 2.5f);
}
