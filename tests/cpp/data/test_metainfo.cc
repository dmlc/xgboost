// Copyright by Contributors
#include <dmlc/io.h>
#include <dmlc/filesystem.h>
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

  EXPECT_EQ(info.labels_.Size(), 0);
  info.SetInfo("label", double2, xgboost::kFloat32, 2);
  EXPECT_EQ(info.labels_.Size(), 2);

  float float2[2] = {1.0f, 2.0f};
  EXPECT_EQ(info.GetWeight(1), 1.0f)
    << "When no weights are given, was expecting default value 1";
  info.SetInfo("weight", float2, xgboost::kFloat32, 2);
  EXPECT_EQ(info.GetWeight(1), 2.0f);

  uint32_t uint32_t2[2] = {1U, 2U};
  EXPECT_EQ(info.base_margin_.Size(), 0);
  info.SetInfo("base_margin", uint32_t2, xgboost::kUInt32, 2);
  EXPECT_EQ(info.base_margin_.Size(), 2);

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

  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/metainfo.binary";
  dmlc::Stream* fs = dmlc::Stream::Create(tmp_file.c_str(), "w");
  info.SaveBinary(fs);
  delete fs;

  ASSERT_EQ(GetFileSize(tmp_file), 84)
    << "Expected saved binary file size to be same as object size";

  fs = dmlc::Stream::Create(tmp_file.c_str(), "r");
  xgboost::MetaInfo inforead;
  inforead.LoadBinary(fs);
  EXPECT_EQ(inforead.labels_.HostVector(), info.labels_.HostVector());
  EXPECT_EQ(inforead.num_col_, info.num_col_);
  EXPECT_EQ(inforead.num_row_, info.num_row_);
  delete fs;
}

TEST(MetaInfo, LoadQid) {
  dmlc::TemporaryDirectory tempdir;
  std::string tmp_file = tempdir.path + "/qid_test.libsvm";
  {
    std::unique_ptr<dmlc::Stream> fs(
      dmlc::Stream::Create(tmp_file.c_str(), "w"));
    dmlc::ostream os(fs.get());
    os << R"qid(3 qid:1 1:1 2:1 3:0 4:0.2 5:0
                2 qid:1 1:0 2:0 3:1 4:0.1 5:1
                1 qid:1 1:0 2:1 3:0 4:0.4 5:0
                1 qid:1 1:0 2:0 3:1 4:0.3 5:0
                1 qid:2 1:0 2:0 3:1 4:0.2 5:0
                2 qid:2 1:1 2:0 3:1 4:0.4 5:0
                1 qid:2 1:0 2:0 3:1 4:0.1 5:0
                1 qid:2 1:0 2:0 3:1 4:0.2 5:0
                2 qid:3 1:0 2:0 3:1 4:0.1 5:1
                3 qid:3 1:1 2:1 3:0 4:0.3 5:0
                4 qid:3 1:1 2:0 3:0 4:0.4 5:1
                1 qid:3 1:0 2:1 3:1 4:0.5 5:0)qid";
    os.set_stream(nullptr);
  }
  std::unique_ptr<xgboost::DMatrix> dmat(
    xgboost::DMatrix::Load(tmp_file, true, false, "libsvm"));

  const xgboost::MetaInfo& info = dmat->Info();
  const std::vector<uint64_t> expected_qids{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  const std::vector<xgboost::bst_uint> expected_group_ptr{0, 4, 8, 12};
  CHECK(info.qids_ == expected_qids);
  CHECK(info.group_ptr_ == expected_group_ptr);
  CHECK_GE(info.kVersion, info.kVersionQidAdded);

  const std::vector<size_t> expected_offset{
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60
  };
  const std::vector<xgboost::Entry> expected_data{
      xgboost::Entry(1, 1),   xgboost::Entry(2, 1),   xgboost::Entry(3, 0),
      xgboost::Entry(4, 0.2), xgboost::Entry(5, 0),   xgboost::Entry(1, 0),
      xgboost::Entry(2, 0),   xgboost::Entry(3, 1),   xgboost::Entry(4, 0.1),
      xgboost::Entry(5, 1),   xgboost::Entry(1, 0),   xgboost::Entry(2, 1),
      xgboost::Entry(3, 0),   xgboost::Entry(4, 0.4), xgboost::Entry(5, 0),
      xgboost::Entry(1, 0),   xgboost::Entry(2, 0),   xgboost::Entry(3, 1),
      xgboost::Entry(4, 0.3), xgboost::Entry(5, 0),   xgboost::Entry(1, 0),
      xgboost::Entry(2, 0),   xgboost::Entry(3, 1),   xgboost::Entry(4, 0.2),
      xgboost::Entry(5, 0),   xgboost::Entry(1, 1),   xgboost::Entry(2, 0),
      xgboost::Entry(3, 1),   xgboost::Entry(4, 0.4), xgboost::Entry(5, 0),
      xgboost::Entry(1, 0),   xgboost::Entry(2, 0),   xgboost::Entry(3, 1),
      xgboost::Entry(4, 0.1), xgboost::Entry(5, 0),   xgboost::Entry(1, 0),
      xgboost::Entry(2, 0),   xgboost::Entry(3, 1),   xgboost::Entry(4, 0.2),
      xgboost::Entry(5, 0),   xgboost::Entry(1, 0),   xgboost::Entry(2, 0),
      xgboost::Entry(3, 1),   xgboost::Entry(4, 0.1), xgboost::Entry(5, 1),
      xgboost::Entry(1, 1),   xgboost::Entry(2, 1),   xgboost::Entry(3, 0),
      xgboost::Entry(4, 0.3), xgboost::Entry(5, 0),   xgboost::Entry(1, 1),
      xgboost::Entry(2, 0),   xgboost::Entry(3, 0),   xgboost::Entry(4, 0.4),
      xgboost::Entry(5, 1),   xgboost::Entry(1, 0),   xgboost::Entry(2, 1),
      xgboost::Entry(3, 1),   xgboost::Entry(4, 0.5), {5, 0}};
  for (const auto &batch : dmat->GetRowBatches()) {
    CHECK_EQ(batch.base_rowid, 0);
    CHECK(batch.offset.HostVector() == expected_offset);
    CHECK(batch.data.HostVector() == expected_data);
  }
}
