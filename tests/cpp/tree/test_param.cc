/**
 * Copyright 2014-2026, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/linalg.h>  // for Vector

#include <sstream>  // for stringstream
#include <vector>   // for vector

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
}  // namespace xgboost::tree
