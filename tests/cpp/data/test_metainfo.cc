// Copyright by Contributors
#include <dmlc/io.h>
#include <dmlc/filesystem.h>
#include <xgboost/data.h>
#include <string>
#include <memory>
#include "../../../src/data/simple_csr_source.h"
#include "../../../src/common/version.h"

#include "../helpers.h"

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
  double vals[2] = {1.0, 2.0};
  info.SetInfo("label", vals, xgboost::DataType::kDouble, 2);
  info.num_row_ = 2;
  info.num_col_ = 1;

  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/metainfo.binary";
  {
    std::unique_ptr<dmlc::Stream> fs {
      dmlc::Stream::Create(tmp_file.c_str(), "w")
    };
    info.SaveBinary(fs.get());
  }

  {
    // Inspect content of header
    std::unique_ptr<dmlc::Stream> fs{
      dmlc::Stream::Create(tmp_file.c_str(), "r")
    };
    auto version = xgboost::Version::Load(fs.get());
    EXPECT_TRUE(xgboost::Version::Same(version));
    uint64_t num_field;
    EXPECT_TRUE(fs->Read(&num_field));
    EXPECT_TRUE(num_field == xgboost::MetaInfo::kNumField);

    std::string field_name;
    xgboost::DataType field_type;

    const std::vector<std::pair<std::string, xgboost::DataType>> expected_fields{
      {u8"num_row", xgboost::DataType::kUInt64},
      {u8"num_col", xgboost::DataType::kUInt64},
      {u8"num_nonzero", xgboost::DataType::kUInt64},
      {u8"labels", xgboost::DataType::kFloat32},
      {u8"group_ptr", xgboost::DataType::kUInt32},
      {u8"weights", xgboost::DataType::kFloat32},
      {u8"base_margin", xgboost::DataType::kFloat32}
    };

    for (uint64_t i = 0; i < num_field; ++i) {
      EXPECT_TRUE(fs->Read(&field_name));
      EXPECT_EQ(field_name, expected_fields[i].first);
      EXPECT_TRUE(fs->Read(&field_type));
      EXPECT_EQ(field_type, expected_fields[i].second);
      switch (field_type) {
        case xgboost::DataType::kFloat32: {
          std::vector<float> vec;
          EXPECT_TRUE(fs->Read(&vec));
          break;
        }
        case xgboost::DataType::kDouble: {
          std::vector<double> vec;
          EXPECT_TRUE(fs->Read(&vec));
          break;
        }
        case xgboost::DataType::kUInt32: {
          std::vector<uint32_t> vec;
          EXPECT_TRUE(fs->Read(&vec));
          break;
        }
        case xgboost::DataType::kUInt64: {
          std::vector<uint64_t> vec;
          EXPECT_TRUE(fs->Read(&vec));
          break;
        }
        default:
          LOG(FATAL) << "Unknown data type" << static_cast<uint8_t>(field_type);
      }
    }
  }

  {
    // Round-trip test
    std::unique_ptr<dmlc::Stream> fs{
      dmlc::Stream::Create(tmp_file.c_str(), "r")
    };
    xgboost::MetaInfo inforead;
    inforead.LoadBinary(fs.get());
    EXPECT_EQ(inforead.num_row_, info.num_row_);
    EXPECT_EQ(inforead.num_col_, info.num_col_);
    EXPECT_EQ(inforead.num_nonzero_, info.num_nonzero_);
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
