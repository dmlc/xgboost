// Copyright by Contributors
#include <xgboost/data.h>
#include <gtest/gtest.h>

TEST(MetaInfo, GetSet) {
  xgboost::MetaInfo info;

  double double2[2] = {1.0, 2.0};
  EXPECT_EQ(info.GetRoot(1), 0)
    << "When no root_index is given, was expecting default value 0";
  info.SetInfo("root_index", double2, xgboost::kDouble, 2);
  EXPECT_EQ(info.GetRoot(1), 2.0f);

  EXPECT_EQ(info.labels.size(), 0);
  info.SetInfo("label", double2, xgboost::kFloat32, 2);
  EXPECT_EQ(info.labels.size(), 2);

  float float2[2] = {1.0f, 2.0f};
  EXPECT_EQ(info.GetWeight(1), 1.0f)
    << "When no weights are given, was expecting default value 1";
  info.SetInfo("weight", float2, xgboost::kFloat32, 2);
  EXPECT_EQ(info.GetWeight(1), 2.0f);

  uint32_t uint32_t2[2] = {1U, 2U};
  EXPECT_EQ(info.base_margin.size(), 0);
  info.SetInfo("base_margin", uint32_t2, xgboost::kUInt32, 2);
  EXPECT_EQ(info.base_margin.size(), 2);

  uint64_t uint64_t2[2] = {1U, 2U};
  EXPECT_EQ(info.group_ptr.size(), 0);
  info.SetInfo("group", uint64_t2, xgboost::kUInt64, 2);
  ASSERT_EQ(info.group_ptr.size(), 3);
  EXPECT_EQ(info.group_ptr[2], 3);
}
