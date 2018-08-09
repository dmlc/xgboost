// Copyright by Contributors
#include <dmlc/io.h>
#include <xgboost/data.h>
#include <string>
#include <memory>
#include "../../../src/data/simple_csr_source.h"

#include "../helpers.h"

TEST(MetaInfo, GetSet) {
  xgboost::MetaInfo info;

  double double2[2] = {1.0, 2.0};
  EXPECT_EQ(info.GetRoot(1), 0)
    << "When no root_index is given, was expecting default value 0";
  info.SetInfo("root_index", double2, xgboost::kDouble, 2);
  EXPECT_EQ(info.GetRoot(1), 2.0f);

  EXPECT_EQ(info.labels_.size(), 0);
  info.SetInfo("label", double2, xgboost::kFloat32, 2);
  EXPECT_EQ(info.labels_.size(), 2);

  float float2[2] = {1.0f, 2.0f};
  EXPECT_EQ(info.GetWeight(1), 1.0f)
    << "When no weights are given, was expecting default value 1";
  info.SetInfo("weight", float2, xgboost::kFloat32, 2);
  EXPECT_EQ(info.GetWeight(1), 2.0f);

  uint32_t uint32_t2[2] = {1U, 2U};
  EXPECT_EQ(info.base_margin_.size(), 0);
  info.SetInfo("base_margin", uint32_t2, xgboost::kUInt32, 2);
  EXPECT_EQ(info.base_margin_.size(), 2);

  uint64_t uint64_t2[2] = {1U, 2U};
  EXPECT_EQ(info.group_ptr_.size(), 0);
  info.SetInfo("group", uint64_t2, xgboost::kUInt64, 2);
  ASSERT_EQ(info.group_ptr_.size(), 3);
  EXPECT_EQ(info.group_ptr_[2], 3);

  info.Clear();
  ASSERT_EQ(info.group_ptr_.size(), 0);
}

TEST(MetaInfo, SaveLoadBinary) {
  xgboost::MetaInfo info;
  double vals[2] = {1.0, 2.0};
  info.SetInfo("label", vals, xgboost::kDouble, 2);
  info.num_row_ = 2;
  info.num_col_ = 1;

  std::string tmp_file = TempFileName();
  dmlc::Stream * fs = dmlc::Stream::Create(tmp_file.c_str(), "w");
  info.SaveBinary(fs);
  delete fs;

  ASSERT_EQ(GetFileSize(tmp_file), 84)
    << "Expected saved binary file size to be same as object size";

  fs = dmlc::Stream::Create(tmp_file.c_str(), "r");
  xgboost::MetaInfo inforead;
  inforead.LoadBinary(fs);
  EXPECT_EQ(inforead.labels_, info.labels_);
  EXPECT_EQ(inforead.num_col_, info.num_col_);
  EXPECT_EQ(inforead.num_row_, info.num_row_);

  std::remove(tmp_file.c_str());
}

TEST(MetaInfo, LoadQid) {
}
