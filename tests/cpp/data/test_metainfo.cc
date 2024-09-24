/**
 * Copyright 2016-2024, XGBoost contributors
 */
#include "test_metainfo.h"

#include <dmlc/io.h>
#include <gmock/gmock.h>
#include <xgboost/data.h>

#include <memory>
#include <string>

#include "../collective/test_worker.h"  // for TestDistributedGlobal
#include "../filesystem.h"              // dmlc::TemporaryDirectory
#include "../helpers.h"                 // for GMockTHrow
#include "xgboost/base.h"

namespace xgboost {
TEST(MetaInfo, GetSet) {
  xgboost::Context ctx;
  xgboost::MetaInfo info;

  double double2[2] = {1.0, 2.0};

  EXPECT_EQ(info.labels.Size(), 0);
  info.SetInfo(ctx, "label", Make1dInterfaceTest(double2, 2));
  EXPECT_EQ(info.labels.Size(), 2);

  float float2[2] = {1.0f, 2.0f};
  EXPECT_EQ(info.GetWeight(1), 1.0f) << "When no weights are given, was expecting default value 1";
  info.SetInfo(ctx, "weight", Make1dInterfaceTest(float2, 2));
  EXPECT_EQ(info.GetWeight(1), 2.0f);

  uint32_t uint32_t2[2] = {1U, 2U};
  EXPECT_EQ(info.base_margin_.Size(), 0);
  info.SetInfo(ctx, "base_margin", Make1dInterfaceTest(uint32_t2, 2));
  EXPECT_EQ(info.base_margin_.Size(), 2);

  uint64_t uint64_t2[2] = {1U, 2U};
  EXPECT_EQ(info.group_ptr_.size(), 0);
  info.SetInfo(ctx, "group", Make1dInterfaceTest(uint64_t2, 2));
  ASSERT_EQ(info.group_ptr_.size(), 3);
  EXPECT_EQ(info.group_ptr_[2], 3);

  info.Clear();
  ASSERT_EQ(info.group_ptr_.size(), 0);
}

TEST(MetaInfo, GetSetFeature) {
  xgboost::MetaInfo info;
  ASSERT_THAT([&] { info.SetFeatureInfo("", nullptr, 0); },
              GMockThrow("Unknown feature info name"));
  EXPECT_THROW(info.SetFeatureInfo("", nullptr, 0), dmlc::Error);
  EXPECT_THROW(info.SetFeatureInfo("foo", nullptr, 0), dmlc::Error);
  EXPECT_NO_THROW(info.SetFeatureInfo("feature_name", nullptr, 0));
  EXPECT_NO_THROW(info.SetFeatureInfo("feature_type", nullptr, 0));
  ASSERT_EQ(info.feature_type_names.size(), 0);
  ASSERT_EQ(info.feature_types.Size(), 0);
  ASSERT_EQ(info.feature_names.size(), 0);

  size_t constexpr kCols = 19;
  std::vector<std::string> types(kCols, u8"float");
  std::vector<char const*> c_types(kCols);
  std::transform(types.cbegin(), types.cend(), c_types.begin(),
                 [](auto const &str) { return str.c_str(); });
  info.num_col_ = 1;
  EXPECT_THROW(
      info.SetFeatureInfo(u8"feature_type", c_types.data(), c_types.size()),
      dmlc::Error);
  info.num_col_ = kCols;
  EXPECT_NO_THROW(
      info.SetFeatureInfo(u8"feature_type", c_types.data(), c_types.size()));

  // Test clear.
  info.SetFeatureInfo("feature_type", nullptr, 0);
  ASSERT_EQ(info.feature_type_names.size(), 0);
  ASSERT_EQ(info.feature_types.Size(), 0);
  // Other conditions are tested in `SaveLoadBinary`.
}

namespace {
void VerifyGetSetFeatureColumnSplit() {
  xgboost::MetaInfo info;
  info.data_split_mode = DataSplitMode::kCol;
  auto const world_size = collective::GetWorldSize();

  auto constexpr kCols{2};
  std::vector<std::string> types{u8"float", u8"c"};
  std::vector<char const *> c_types(kCols);
  std::transform(types.cbegin(), types.cend(), c_types.begin(),
                 [](auto const &str) { return str.c_str(); });
  info.num_col_ = kCols;
  ASSERT_THAT([&] { info.SetFeatureInfo(u8"feature_type", c_types.data(), c_types.size()); },
              GMockThrow("Length of feature_type must be equal to number of columns"));
  info.num_col_ = kCols * world_size;
  EXPECT_NO_THROW(info.SetFeatureInfo(u8"feature_type", c_types.data(), c_types.size()));
  std::vector<std::string> expected_type_names{u8"float", u8"c",     u8"float",
                                               u8"c",     u8"float", u8"c"};
  EXPECT_EQ(info.feature_type_names, expected_type_names);
  std::vector<xgboost::FeatureType> expected_types{
      xgboost::FeatureType::kNumerical, xgboost::FeatureType::kCategorical,
      xgboost::FeatureType::kNumerical, xgboost::FeatureType::kCategorical,
      xgboost::FeatureType::kNumerical, xgboost::FeatureType::kCategorical};
  EXPECT_EQ(info.feature_types.HostVector(), expected_types);

  std::vector<std::string> names{u8"feature0", u8"feature1"};
  std::vector<char const *> c_names(kCols);
  std::transform(names.cbegin(), names.cend(), c_names.begin(),
                 [](auto const &str) { return str.c_str(); });
  info.num_col_ = kCols;
  ASSERT_THAT([&] { info.SetFeatureInfo(u8"feature_name", c_names.data(), c_names.size()); },
              GMockThrow("Length of feature_name must be equal to number of columns"));
  info.num_col_ = kCols * world_size;
  EXPECT_NO_THROW(info.SetFeatureInfo(u8"feature_name", c_names.data(), c_names.size()));
  std::vector<std::string> expected_names{u8"0.feature0", u8"0.feature1", u8"1.feature0",
                                          u8"1.feature1", u8"2.feature0", u8"2.feature1"};
  EXPECT_EQ(info.feature_names, expected_names);
}
}  // anonymous namespace

TEST(MetaInfo, GetSetFeatureColumnSplit) {
  auto constexpr kWorkers{3};
  collective::TestDistributedGlobal(kWorkers, VerifyGetSetFeatureColumnSplit);
}

TEST(MetaInfo, SaveLoadBinary) {
  xgboost::MetaInfo info;
  xgboost::Context ctx;

  uint64_t constexpr kRows { 64 }, kCols { 32 };
  auto generator = []() {
                     static float f = 0;
                     return f++;
                   };
  std::vector<float> values (kRows);
  std::generate(values.begin(), values.end(), generator);
  info.SetInfo(ctx, "label", Make1dInterfaceTest(values.data(), kRows));
  info.SetInfo(ctx, "weight", Make1dInterfaceTest(values.data(), kRows));
  info.SetInfo(ctx, "base_margin", Make1dInterfaceTest(values.data(), kRows));

  info.num_row_ = kRows;
  info.num_col_ = kCols;

  auto featname = u8"特征名";
  std::vector<std::string> types(kCols, u8"float");
  std::vector<char const*> c_types(kCols);
  std::transform(types.cbegin(), types.cend(), c_types.begin(),
                 [](auto const &str) { return str.c_str(); });
  info.SetFeatureInfo(u8"feature_type", c_types.data(), c_types.size());
  std::vector<std::string> names(kCols, featname);
  std::vector<char const*> c_names(kCols);
  std::transform(names.cbegin(), names.cend(), c_names.begin(),
                 [](auto const &str) { return str.c_str(); });
  info.SetFeatureInfo(u8"feature_name", c_names.data(), c_names.size());;

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

    ASSERT_EQ(inforead.labels.Data()->HostVector(), values);
    EXPECT_EQ(inforead.labels.Data()->HostVector(), info.labels.Data()->HostVector());
    EXPECT_EQ(inforead.group_ptr_, info.group_ptr_);
    EXPECT_EQ(inforead.weights_.HostVector(), info.weights_.HostVector());

    auto orig_margin = info.base_margin_.View(xgboost::DeviceOrd::CPU());
    auto read_margin = inforead.base_margin_.View(xgboost::DeviceOrd::CPU());
    EXPECT_TRUE(std::equal(orig_margin.Values().cbegin(), orig_margin.Values().cend(),
                           read_margin.Values().cbegin()));

    EXPECT_EQ(inforead.feature_type_names.size(), kCols);
    EXPECT_EQ(inforead.feature_types.Size(), kCols);
    EXPECT_TRUE(std::all_of(inforead.feature_type_names.cbegin(),
                            inforead.feature_type_names.cend(),
                            [](auto const &str) { return str == u8"float"; }));
    auto h_ft = inforead.feature_types.HostSpan();
    EXPECT_TRUE(std::all_of(h_ft.cbegin(), h_ft.cend(), [](auto f) {
      return f == xgboost::FeatureType::kNumerical;
    }));

    EXPECT_EQ(inforead.feature_names.size(), kCols);
    EXPECT_TRUE(std::all_of(inforead.feature_names.cbegin(),
                            inforead.feature_names.cend(),
                            [=](auto const& str) {
                              return str == featname;
                            }));
  }
}

TEST(MetaInfo, LoadQid) {
  dmlc::TemporaryDirectory tempdir;
  std::string tmp_file = tempdir.path + "/qid_test.libsvm";
  {
    std::unique_ptr<dmlc::Stream> fs(dmlc::Stream::Create(tmp_file.c_str(), "w"));
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
      xgboost::DMatrix::Load(tmp_file + "?format=libsvm", true, xgboost::DataSplitMode::kRow));

  const xgboost::MetaInfo& info = dmat->Info();
  const std::vector<xgboost::bst_uint> expected_group_ptr{0, 4, 8, 12};
  CHECK(info.group_ptr_ == expected_group_ptr);

  const std::vector<xgboost::bst_idx_t> expected_offset{
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

TEST(MetaInfo, CPUQid) {
  xgboost::MetaInfo info;
  xgboost::Context ctx;
  info.num_row_ = 100;
  std::vector<uint32_t> qid(info.num_row_, 0);
  for (size_t i = 0; i < qid.size(); ++i) {
    qid[i] = i;
  }

  info.SetInfo(ctx, "qid", Make1dInterfaceTest(qid.data(), info.num_row_));
  ASSERT_EQ(info.group_ptr_.size(), info.num_row_ + 1);
  ASSERT_EQ(info.group_ptr_.front(), 0);
  ASSERT_EQ(info.group_ptr_.back(), info.num_row_);

  for (size_t i = 0; i < info.num_row_ + 1; ++i) {
    ASSERT_EQ(info.group_ptr_[i], i);
  }
}

TEST(MetaInfo, Validate) {
  xgboost::MetaInfo info;
  info.num_row_ = 10;
  info.num_nonzero_ = 12;
  info.num_col_ = 3;
  std::vector<xgboost::bst_group_t> groups (11);
  Context ctx;
  info.SetInfo(ctx, "group", Make1dInterfaceTest(groups.data(), groups.size()));
  EXPECT_THROW(info.Validate(FstCU()), dmlc::Error);

  std::vector<float> labels(info.num_row_ + 1);
  EXPECT_THROW(
      { info.SetInfo(ctx, "label", Make1dInterfaceTest(labels.data(), info.num_row_ + 1)); },
      dmlc::Error);

  // Make overflow data, which can happen when users pass group structure as int
  // or float.
  groups = {};
  for (size_t i = 0; i < 63; ++i) {
    groups.push_back(1562500);
  }
  groups.push_back(static_cast<xgboost::bst_group_t>(-1));
  EXPECT_THROW(info.SetInfo(ctx, "group", Make1dInterfaceTest(groups.data(), groups.size())),
               dmlc::Error);

#if defined(XGBOOST_USE_CUDA)
  info.group_ptr_.clear();
  labels.resize(info.num_row_);
  info.SetInfo(ctx, "label", Make1dInterfaceTest(labels.data(), info.num_row_));
  info.labels.SetDevice(FstCU());
  EXPECT_THROW(info.Validate(DeviceOrd::CUDA(1)), dmlc::Error);

  xgboost::HostDeviceVector<xgboost::bst_group_t> d_groups{groups};
  d_groups.SetDevice(FstCU());
  d_groups.DevicePointer();  // pull to device
  std::string arr_interface_str{ArrayInterfaceStr(xgboost::linalg::MakeVec(
      d_groups.ConstDevicePointer(), d_groups.Size(), xgboost::DeviceOrd::CUDA(0)))};
  EXPECT_THROW(info.SetInfo(ctx, "group", xgboost::StringView{arr_interface_str}), dmlc::Error);
#endif  // defined(XGBOOST_USE_CUDA)
}

TEST(MetaInfo, HostExtend) {
  xgboost::MetaInfo lhs, rhs;
  xgboost::Context ctx;
  size_t const kRows = 100;
  lhs.labels.Reshape(kRows);
  lhs.num_row_ = kRows;
  rhs.labels.Reshape(kRows);
  rhs.num_row_ = kRows;
  ASSERT_TRUE(lhs.labels.Data()->HostCanRead());
  ASSERT_TRUE(rhs.labels.Data()->HostCanRead());

  size_t per_group = 10;
  std::vector<xgboost::bst_group_t> groups;
  for (size_t g = 0; g < kRows / per_group; ++g) {
    groups.emplace_back(per_group);
  }
  lhs.SetInfo(ctx, "group", Make1dInterfaceTest(groups.data(), groups.size()));
  rhs.SetInfo(ctx, "group", Make1dInterfaceTest(groups.data(), groups.size()));

  lhs.Extend(rhs, true, true);
  ASSERT_EQ(lhs.num_row_, kRows * 2);
  ASSERT_TRUE(lhs.labels.Data()->HostCanRead());
  ASSERT_TRUE(rhs.labels.Data()->HostCanRead());
  ASSERT_FALSE(lhs.labels.Data()->DeviceCanRead());
  ASSERT_FALSE(rhs.labels.Data()->DeviceCanRead());

  ASSERT_EQ(lhs.group_ptr_.front(), 0);
  ASSERT_EQ(lhs.group_ptr_.back(), kRows * 2);
  for (size_t i = 0; i < kRows * 2 / per_group; ++i) {
    ASSERT_EQ(lhs.group_ptr_.at(i), per_group * i);
  }
}

TEST(MetaInfo, CPUStridedData) { TestMetaInfoStridedData(DeviceOrd::CPU()); }

namespace {
class TestMetaInfo : public ::testing::TestWithParam<std::tuple<bst_target_t, bool>> {
 public:
  void Run(Context const *ctx, bst_target_t n_targets) {
    MetaInfo info;
    info.num_row_ = 128;
    info.num_col_ = 3;
    info.feature_names.resize(info.num_col_, "a");
    info.labels.Reshape(info.num_row_, n_targets);

    HostDeviceVector<bst_idx_t> ridx(info.num_row_ / 2, 0);
    ridx.SetDevice(ctx->Device());
    auto h_ridx = ridx.HostSpan();
    for (std::size_t i = 0, j = 0; i < ridx.Size(); i++, j += 2) {
      h_ridx[i] = j;
    }

    {
      info.weights_.Resize(info.num_row_);
      auto h_w = info.weights_.HostSpan();
      std::iota(h_w.begin(), h_w.end(), 0);
    }

    auto out = info.Slice(ctx, ctx->IsCPU() ? h_ridx : ridx.ConstDeviceSpan(), /*nnz=*/256);

    ASSERT_EQ(info.labels.Device(), ctx->Device());
    auto h_y = info.labels.HostView();
    auto h_y_out = out.labels.HostView();
    ASSERT_EQ(h_y_out.Shape(0), ridx.Size());
    ASSERT_EQ(h_y_out.Shape(1), n_targets);

    auto h_w = info.weights_.ConstHostSpan();
    auto h_w_out = out.weights_.ConstHostSpan();
    ASSERT_EQ(h_w_out.size(), ridx.Size());

    for (std::size_t i = 0; i < ridx.Size(); ++i) {
      for (bst_target_t t = 0; t < n_targets; ++t) {
        ASSERT_EQ(h_y_out(i, t), h_y(h_ridx[i], t));
      }
      ASSERT_EQ(h_w_out[i], h_w[h_ridx[i]]);
    }

    for (auto v : info.feature_names) {
      ASSERT_EQ(v, "a");
    }
  }
};
}  // anonymous namespace

TEST_P(TestMetaInfo, Slice) {
  Context ctx;
  auto [n_targets, is_cuda] = this->GetParam();
  if (is_cuda) {
    ctx = MakeCUDACtx(0);
  }
  this->Run(&ctx, n_targets);
}

INSTANTIATE_TEST_SUITE_P(Cpu, TestMetaInfo,
                         ::testing::Values(std::tuple{1u, false}, std::tuple{3u, false}));

#if defined(XGBOOST_USE_CUDA)
INSTANTIATE_TEST_SUITE_P(Gpu, TestMetaInfo,
                         ::testing::Values(std::tuple{1u, true}, std::tuple{3u, true}));
#endif  // defined(XGBOOST_USE_CUDA)
}  // namespace xgboost
