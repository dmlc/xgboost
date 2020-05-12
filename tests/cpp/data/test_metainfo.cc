// Copyright 2016-2020 by Contributors
#include <dmlc/io.h>
#include <dmlc/filesystem.h>
#include <xgboost/data.h>
#include <string>
#include <memory>
#include "../../../src/common/version.h"

#include "../helpers.h"
#include "xgboost/base.h"

TEST(MetaInfo, GetSet) {
  xgboost::MetaInfo info;

  double double2[2] = {1.0, 2.0};

  EXPECT_EQ(info.labels_.Size(), 0);
  info.SetInfo("label", double2, xgboost::DataType::kFloat32, 2);
  EXPECT_EQ(info.labels_.Size(), 2);

  float float2[2] = {1.0f, 2.0f};
  EXPECT_EQ(info.GetWeight(1), 1.0f)
    << "When no weights are given, was expecting default value 1";
  info.SetInfo("weight", float2, xgboost::DataType::kFloat32, 2);
  EXPECT_EQ(info.GetWeight(1), 2.0f);

  uint32_t uint32_t2[2] = {1U, 2U};
  EXPECT_EQ(info.base_margin_.Size(), 0);
  info.SetInfo("base_margin", uint32_t2, xgboost::DataType::kUInt32, 2);
  EXPECT_EQ(info.base_margin_.Size(), 2);

  uint64_t uint64_t2[2] = {1U, 2U};
  EXPECT_EQ(info.group_ptr_.size(), 0);
  info.SetInfo("group", uint64_t2, xgboost::DataType::kUInt64, 2);
  ASSERT_EQ(info.group_ptr_.size(), 3);
  EXPECT_EQ(info.group_ptr_[2], 3);

  info.Clear();
  ASSERT_EQ(info.group_ptr_.size(), 0);
}

TEST(MetaInfo, SaveLoadBinary) {
  xgboost::MetaInfo info;
  uint64_t constexpr kRows { 64 }, kCols { 32 };
  auto generator = []() {
                     static float f = 0;
                     return f++;
                   };
  std::vector<float> values (kRows);
  std::generate(values.begin(), values.end(), generator);
  info.SetInfo("label", values.data(), xgboost::DataType::kFloat32, kRows);
  info.SetInfo("weight", values.data(), xgboost::DataType::kFloat32, kRows);
  info.SetInfo("base_margin", values.data(), xgboost::DataType::kFloat32, kRows);
  info.num_row_ = kRows;
  info.num_col_ = kCols;

  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/metainfo.binary";
  {
    std::unique_ptr<dmlc::Stream> fs {
      dmlc::Stream::Create(tmp_file.c_str(), "w")
    };
    info.SaveBinary(fs.get());
  }

  {
    // Round-trip test
    std::unique_ptr<dmlc::Stream> fs {
      dmlc::Stream::Create(tmp_file.c_str(), "r")
    };
    xgboost::MetaInfo inforead;
    inforead.LoadBinary(fs.get());
    ASSERT_EQ(inforead.num_row_, kRows);
    EXPECT_EQ(inforead.num_row_, info.num_row_);
    EXPECT_EQ(inforead.num_col_, info.num_col_);
    EXPECT_EQ(inforead.num_nonzero_, info.num_nonzero_);

    ASSERT_EQ(inforead.labels_.HostVector(), values);
    EXPECT_EQ(inforead.labels_.HostVector(), info.labels_.HostVector());
    EXPECT_EQ(inforead.group_ptr_, info.group_ptr_);
    EXPECT_EQ(inforead.weights_.HostVector(), info.weights_.HostVector());
    EXPECT_EQ(inforead.base_margin_.HostVector(), info.base_margin_.HostVector());
  }
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
  const std::vector<xgboost::bst_uint> expected_group_ptr{0, 4, 8, 12};
  CHECK(info.group_ptr_ == expected_group_ptr);

  const std::vector<xgboost::bst_row_t> expected_offset{
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
  for (const auto &batch : dmat->GetBatches<xgboost::SparsePage>()) {
    CHECK_EQ(batch.base_rowid, 0);
    CHECK(batch.offset.HostVector() == expected_offset);
    CHECK(batch.data.HostVector() == expected_data);
  }
}

TEST(MetaInfo, Validate) {
  xgboost::MetaInfo info;
  info.num_row_ = 10;
  info.num_nonzero_ = 12;
  info.num_col_ = 3;
  std::vector<xgboost::bst_group_t> groups (11);
  info.SetInfo("group", groups.data(), xgboost::DataType::kUInt32, 11);
  EXPECT_THROW(info.Validate(0), dmlc::Error);

  std::vector<float> labels(info.num_row_ + 1);
  info.SetInfo("label", labels.data(), xgboost::DataType::kFloat32, info.num_row_ + 1);
  EXPECT_THROW(info.Validate(0), dmlc::Error);

#if defined(XGBOOST_USE_CUDA)
  info.group_ptr_.clear();
  labels.resize(info.num_row_);
  info.SetInfo("label", labels.data(), xgboost::DataType::kFloat32, info.num_row_);
  info.labels_.SetDevice(0);
  EXPECT_THROW(info.Validate(1), dmlc::Error);
#endif  // defined(XGBOOST_USE_CUDA)
}
