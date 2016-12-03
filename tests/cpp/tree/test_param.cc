// Copyright by Contributors
#include "../../src/tree/param.h"

#include "../helpers.h"

TEST(Param, VectorIOStream) {
  std::vector<int> vals = {3, 2, 1};
  std::stringstream ss;
  std::vector<int> vals_in;
  
  ss << vals;
  EXPECT_EQ(ss.str(), "(3,2,1)");

  ss >> vals_in;
  EXPECT_EQ(vals_in, vals);

  vals.clear(); ss.flush(); ss.clear(); ss.str("");
  vals = {1};
  ss << vals;
  EXPECT_EQ(ss.str(), "(1,)");
}

TEST(Param, VectorStreamRead) {
  std::vector<int> vals = {3, 2, 1};
  std::stringstream ss;
  std::vector<int> vals_in;

  vals_in.clear(); ss.flush(); ss.clear(); ss.str("");
  ss << "(3, 2, 1)";
  ss >> vals_in;
  EXPECT_EQ(vals_in, vals);

  vals_in.clear(); ss.flush(); ss.clear(); ss.str("");
  ss << "(3L,2L,1L)";
  ss >> vals_in;
  EXPECT_EQ(vals_in, vals);

  vals_in.clear(); ss.flush(); ss.clear(); ss.str("");
  ss << " (3,2,1,)";
  ss >> vals_in;
  EXPECT_EQ(vals_in, vals);

  vals_in.clear(); ss.flush(); ss.clear(); ss.str("");
  ss << " ( 3, 2,1 )";
  ss >> vals_in;
  EXPECT_EQ(vals_in, vals);

  vals_in.clear(); ss.flush(); ss.clear(); ss.str("");
  ss << " ( 3, 2,1 ) ";
  ss >> vals_in;
  EXPECT_EQ(vals_in, vals);

  vals_in.clear(); ss.flush(); ss.clear(); ss.str("");
  ss << " 321 ";
  ss >> vals_in;
  EXPECT_EQ(vals_in[0], 321);

  vals_in.clear(); ss.flush(); ss.clear(); ss.str("");
  ss << "(3.0,2,1)";
  ss >> vals_in;
  EXPECT_NE(vals_in, vals);

  vals_in.clear(); ss.flush(); ss.clear(); ss.str("");
  ss << "1a";
  ss >> vals_in;
  EXPECT_NE(vals_in, vals);

  vals_in.clear(); ss.flush(); ss.clear(); ss.str("");
  ss << "abcde";
  ss >> vals_in;
  EXPECT_NE(vals_in, vals);

  vals_in.clear(); ss.flush(); ss.clear(); ss.str("");
  ss << "(3,2,1";
  ss >> vals_in;
  EXPECT_NE(vals_in, vals);
}
