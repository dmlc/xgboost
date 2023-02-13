/**
 * Copyright 2016-2023 by XGBoost Contributors
 */
#include <xgboost/data.h>

#include <array>   // std::array
#include <limits>  // std::numeric_limits
#include <memory>  // std::unique_ptr

#include "../../../src/data/adapter.h"         // ArrayAdapter
#include "../../../src/data/simple_dmatrix.h"  // SimpleDMatrix
#include "../filesystem.h"                     // dmlc::TemporaryDirectory
#include "../helpers.h"                        // RandomDataGenerator,CreateSimpleTestData
#include "xgboost/base.h"
#include "xgboost/host_device_vector.h"  // HostDeviceVector
#include "xgboost/string_view.h"         // StringView

using namespace xgboost;  // NOLINT

namespace {
std::string UriSVM(std::string name) { return name + "?format=libsvm"; }
}  // namespace

TEST(SimpleDMatrix, MetaInfo) {
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/simple.libsvm";
  CreateSimpleTestData(tmp_file);
  xgboost::DMatrix *dmat = xgboost::DMatrix::Load(UriSVM(tmp_file));

  // Test the metadata that was parsed
  EXPECT_EQ(dmat->Info().num_row_, 2);
  EXPECT_EQ(dmat->Info().num_col_, 5);
  EXPECT_EQ(dmat->Info().num_nonzero_, 6);
  EXPECT_EQ(dmat->Info().labels.Size(), dmat->Info().num_row_);
  EXPECT_EQ(dmat->Info().data_split_mode, DataSplitMode::kRow);

  delete dmat;
}

TEST(SimpleDMatrix, RowAccess) {
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/simple.libsvm";
  CreateSimpleTestData(tmp_file);
  xgboost::DMatrix *dmat = xgboost::DMatrix::Load(UriSVM(tmp_file), false);

  // Loop over the batches and count the records
  int64_t row_count = 0;
  for (auto &batch : dmat->GetBatches<xgboost::SparsePage>()) {
    row_count += batch.Size();
  }
  EXPECT_EQ(row_count, dmat->Info().num_row_);
  // Test the data read into the first row
  auto &batch = *dmat->GetBatches<xgboost::SparsePage>().begin();
  auto page = batch.GetView();
  auto first_row = page[0];
  ASSERT_EQ(first_row.size(), 3);
  EXPECT_EQ(first_row[2].index, 2);
  EXPECT_EQ(first_row[2].fvalue, 20);

  delete dmat;
}

TEST(SimpleDMatrix, ColAccessWithoutBatches) {
  Context ctx;
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/simple.libsvm";
  CreateSimpleTestData(tmp_file);
  xgboost::DMatrix *dmat = xgboost::DMatrix::Load(UriSVM(tmp_file));

  ASSERT_TRUE(dmat->SingleColBlock());

  // Loop over the batches and assert the data is as expected
  int64_t num_col_batch = 0;
  for (const auto &batch : dmat->GetBatches<xgboost::SortedCSCPage>(&ctx)) {
    num_col_batch += 1;
    EXPECT_EQ(batch.Size(), dmat->Info().num_col_)
        << "Expected batch size = number of cells as #batches is 1.";
  }
  EXPECT_EQ(num_col_batch, 1) << "Expected number of batches to be 1";
  delete dmat;
}

TEST(SimpleDMatrix, Empty) {
  std::vector<float> data{};
  std::vector<unsigned> feature_idx = {};
  std::vector<size_t> row_ptr = {};

  data::CSRAdapter csr_adapter(row_ptr.data(), feature_idx.data(), data.data(),
                               0, 0, 0);
  std::unique_ptr<data::SimpleDMatrix> dmat(new data::SimpleDMatrix(
      &csr_adapter, std::numeric_limits<float>::quiet_NaN(), 1));
  CHECK_EQ(dmat->Info().num_nonzero_, 0);
  CHECK_EQ(dmat->Info().num_row_, 0);
  CHECK_EQ(dmat->Info().num_col_, 0);
  for (auto &batch : dmat->GetBatches<SparsePage>()) {
    CHECK_EQ(batch.Size(), 0);
  }

  data::DenseAdapter dense_adapter(nullptr, 0, 0);
  dmat.reset( new data::SimpleDMatrix(&dense_adapter,
                                      std::numeric_limits<float>::quiet_NaN(), 1) );
  CHECK_EQ(dmat->Info().num_nonzero_, 0);
  CHECK_EQ(dmat->Info().num_row_, 0);
  CHECK_EQ(dmat->Info().num_col_, 0);
  for (auto &batch : dmat->GetBatches<SparsePage>()) {
    CHECK_EQ(batch.Size(), 0);
  }

  data::CSCAdapter csc_adapter(nullptr, nullptr, nullptr, 0, 0);
  dmat.reset(new data::SimpleDMatrix(
      &csc_adapter, std::numeric_limits<float>::quiet_NaN(), 1));
  CHECK_EQ(dmat->Info().num_nonzero_, 0);
  CHECK_EQ(dmat->Info().num_row_, 0);
  CHECK_EQ(dmat->Info().num_col_, 0);
  for (auto &batch : dmat->GetBatches<SparsePage>()) {
    CHECK_EQ(batch.Size(), 0);
  }
}

TEST(SimpleDMatrix, MissingData) {
  std::vector<float> data{0.0, std::nanf(""), 1.0};
  std::vector<unsigned> feature_idx = {0, 1, 0};
  std::vector<size_t> row_ptr = {0, 2, 3};

  data::CSRAdapter adapter(row_ptr.data(), feature_idx.data(), data.data(), 2,
                           3, 2);
  std::unique_ptr<data::SimpleDMatrix> dmat{new data::SimpleDMatrix{
      &adapter, std::numeric_limits<float>::quiet_NaN(), 1}};
  CHECK_EQ(dmat->Info().num_nonzero_, 2);
  dmat.reset(new data::SimpleDMatrix(&adapter, 1.0, 1));
  CHECK_EQ(dmat->Info().num_nonzero_, 1);

  {
    data[1] = std::numeric_limits<float>::infinity();
    data::DenseAdapter adapter(data.data(), data.size(), 1);
    EXPECT_THROW(data::SimpleDMatrix dmat(
                     &adapter, std::numeric_limits<float>::quiet_NaN(), -1),
                 dmlc::Error);
  }
}

TEST(SimpleDMatrix, EmptyRow) {
  std::vector<float> data{0.0, 1.0};
  std::vector<unsigned> feature_idx = {0, 1};
  std::vector<size_t> row_ptr = {0, 2, 2};

  data::CSRAdapter adapter(row_ptr.data(), feature_idx.data(), data.data(), 2,
                           2, 2);
  data::SimpleDMatrix dmat(&adapter, std::numeric_limits<float>::quiet_NaN(),
                           1);
  CHECK_EQ(dmat.Info().num_nonzero_, 2);
  CHECK_EQ(dmat.Info().num_row_, 2);
  CHECK_EQ(dmat.Info().num_col_, 2);
}

TEST(SimpleDMatrix, FromDense) {
  int m = 3;
  int n = 2;
  std::vector<float> data = {1, 2, 3, 4, 5, 6};
  data::DenseAdapter adapter(data.data(), m, n);
  data::SimpleDMatrix dmat(&adapter, std::numeric_limits<float>::quiet_NaN(),
                           -1);
  EXPECT_EQ(dmat.Info().num_col_, 2);
  EXPECT_EQ(dmat.Info().num_row_, 3);
  EXPECT_EQ(dmat.Info().num_nonzero_, 6);

  for (auto &batch : dmat.GetBatches<SparsePage>()) {
    auto page = batch.GetView();
    for (auto i = 0ull; i < batch.Size(); i++) {
      auto inst = page[i];
      for (auto j = 0ull; j < inst.size(); j++) {
        EXPECT_EQ(inst[j].fvalue, data[i * n + j]);
        EXPECT_EQ(inst[j].index, j);
      }
    }
  }
}

TEST(SimpleDMatrix, FromCSC) {
  std::vector<float> data = {1, 3, 2, 4, 5};
  std::vector<unsigned> row_idx = {0, 1, 0, 1, 2};
  std::vector<size_t> col_ptr = {0, 2, 5};
  data::CSCAdapter adapter(col_ptr.data(), row_idx.data(), data.data(), 2, 3);
  data::SimpleDMatrix dmat(&adapter, std::numeric_limits<float>::quiet_NaN(),
                           -1);
  EXPECT_EQ(dmat.Info().num_col_, 2);
  EXPECT_EQ(dmat.Info().num_row_, 3);
  EXPECT_EQ(dmat.Info().num_nonzero_, 5);

  auto &batch = *dmat.GetBatches<SparsePage>().begin();
  auto page = batch.GetView();
  auto inst = page[0];
  EXPECT_EQ(inst[0].fvalue, 1);
  EXPECT_EQ(inst[0].index, 0);
  EXPECT_EQ(inst[1].fvalue, 2);
  EXPECT_EQ(inst[1].index, 1);

  inst = page[1];
  EXPECT_EQ(inst[0].fvalue, 3);
  EXPECT_EQ(inst[0].index, 0);
  EXPECT_EQ(inst[1].fvalue, 4);
  EXPECT_EQ(inst[1].index, 1);

  inst = page[2];
  EXPECT_EQ(inst[0].fvalue, 5);
  EXPECT_EQ(inst[0].index, 1);
}

TEST(SimpleDMatrix, FromFile) {
  dmlc::TemporaryDirectory tempdir;
  std::string filename = tempdir.path + "test.libsvm";
  CreateBigTestData(filename, 3 * 5);
  // Add an empty row at the end of the matrix
  {
    std::ofstream fo(filename, std::ios::app | std::ios::out);
    fo << "0\n";
  }
  constexpr size_t kExpectedNumRow = 6;
  std::unique_ptr<dmlc::Parser<uint32_t>> parser(
      dmlc::Parser<uint32_t>::Create(filename.c_str(), 0, 1, "auto"));

  auto verify_batch = [kExpectedNumRow](SparsePage const &page) {
    auto batch = page.GetView();
    EXPECT_EQ(batch.Size(), kExpectedNumRow);
    EXPECT_EQ(page.offset.HostVector(),
              std::vector<bst_row_t>({0, 3, 6, 9, 12, 15, 15}));
    EXPECT_EQ(page.base_rowid, 0);

    for (auto i = 0ull; i < batch.Size() - 1; i++) {
      if (i % 2 == 0) {
        EXPECT_EQ(batch[i][0].index, 0);
        EXPECT_EQ(batch[i][1].index, 1);
        EXPECT_EQ(batch[i][2].index, 2);
      } else {
        EXPECT_EQ(batch[i][0].index, 0);
        EXPECT_EQ(batch[i][1].index, 3);
        EXPECT_EQ(batch[i][2].index, 4);
      }
    }
  };

  constexpr bst_feature_t kCols = 5;
  data::FileAdapter adapter(parser.get());
  data::SimpleDMatrix dmat(&adapter, std::numeric_limits<float>::quiet_NaN(),
                           1);
  ASSERT_EQ(dmat.Info().num_col_, kCols);

  for (auto &batch : dmat.GetBatches<SparsePage>()) {
    verify_batch(batch);
  }
}

TEST(SimpleDMatrix, Slice) {
  size_t constexpr kRows {16};
  size_t constexpr kCols {8};
  size_t constexpr kClasses {3};
  auto p_m = RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix(true);
  auto& weights = p_m->Info().weights_.HostVector();
  weights.resize(kRows);
  std::iota(weights.begin(), weights.end(), 0.0f);

  auto& lower = p_m->Info().labels_lower_bound_.HostVector();
  auto& upper = p_m->Info().labels_upper_bound_.HostVector();
  lower.resize(kRows);
  upper.resize(kRows);

  std::iota(lower.begin(), lower.end(), 0.0f);
  std::iota(upper.begin(), upper.end(), 1.0f);

  auto& margin = p_m->Info().base_margin_;
  margin = decltype(p_m->Info().base_margin_){{kRows, kClasses}, Context::kCpuId};

  std::array<int32_t, 3> ridxs {1, 3, 5};
  std::unique_ptr<DMatrix> out { p_m->Slice(ridxs) };
  ASSERT_EQ(out->Info().labels.Size(), ridxs.size());
  ASSERT_EQ(out->Info().labels_lower_bound_.Size(), ridxs.size());
  ASSERT_EQ(out->Info().labels_upper_bound_.Size(), ridxs.size());
  ASSERT_EQ(out->Info().base_margin_.Size(), ridxs.size() * kClasses);

  for (auto const& in_batch : p_m->GetBatches<SparsePage>()) {
    auto in_page = in_batch.GetView();
    for (auto const &out_batch : out->GetBatches<SparsePage>()) {
      auto out_page = out_batch.GetView();
      for (size_t i = 0; i < ridxs.size(); ++i) {
        auto ridx = ridxs[i];
        auto out_inst = out_page[i];
        auto in_inst = in_page[ridx];
        ASSERT_EQ(out_inst.size(), in_inst.size()) << i;
        for (size_t j = 0; j < in_inst.size(); ++j) {
          ASSERT_EQ(in_inst[j].fvalue, out_inst[j].fvalue);
          ASSERT_EQ(in_inst[j].index, out_inst[j].index);
        }

        ASSERT_EQ(p_m->Info().labels_lower_bound_.HostVector().at(ridx),
                  out->Info().labels_lower_bound_.HostVector().at(i));
        ASSERT_EQ(p_m->Info().labels_upper_bound_.HostVector().at(ridx),
                  out->Info().labels_upper_bound_.HostVector().at(i));
        ASSERT_EQ(p_m->Info().weights_.HostVector().at(ridx),
                  out->Info().weights_.HostVector().at(i));

        auto out_margin = out->Info().base_margin_.View(Context::kCpuId);
        auto in_margin = margin.View(Context::kCpuId);
        for (size_t j = 0; j < kClasses; ++j) {
          ASSERT_EQ(out_margin(i, j), in_margin(ridx, j));
        }
      }
    }
  }

  ASSERT_EQ(out->Info().num_col_, out->Info().num_col_);
  ASSERT_EQ(out->Info().num_row_, ridxs.size());
  ASSERT_EQ(out->Info().num_nonzero_, ridxs.size() * kCols);  // dense

  {
    HostDeviceVector<float> data;
    auto arr_str = RandomDataGenerator{kRows, kCols, 0.0}.GenerateArrayInterface(&data);
    auto adapter = data::ArrayAdapter{StringView{arr_str}};
    auto n_threads = 2;
    std::unique_ptr<DMatrix> p_fmat{
        DMatrix::Create(&adapter, std::numeric_limits<float>::quiet_NaN(), n_threads, "")};
    std::unique_ptr<DMatrix> slice{p_fmat->Slice(ridxs)};
    ASSERT_LE(slice->Ctx()->Threads(), n_threads);
  }
}

TEST(SimpleDMatrix, SliceCol) {
  size_t constexpr kRows {16};
  size_t constexpr kCols {8};
  size_t constexpr kClasses {3};
  auto p_m = RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix(true);
  auto& weights = p_m->Info().weights_.HostVector();
  weights.resize(kRows);
  std::iota(weights.begin(), weights.end(), 0.0f);

  auto& lower = p_m->Info().labels_lower_bound_.HostVector();
  auto& upper = p_m->Info().labels_upper_bound_.HostVector();
  lower.resize(kRows);
  upper.resize(kRows);

  std::iota(lower.begin(), lower.end(), 0.0f);
  std::iota(upper.begin(), upper.end(), 1.0f);

  auto& margin = p_m->Info().base_margin_;
  margin = decltype(p_m->Info().base_margin_){{kRows, kClasses}, Context::kCpuId};

  auto constexpr kSlices {2};
  auto constexpr kSliceSize {4};
  for (auto slice = 0; slice < kSlices; slice++) {
    std::unique_ptr<DMatrix> out { p_m->SliceCol(kSlices, slice) };
    ASSERT_EQ(out->Info().labels.Size(), kRows);
    ASSERT_EQ(out->Info().labels_lower_bound_.Size(), kRows);
    ASSERT_EQ(out->Info().labels_upper_bound_.Size(), kRows);
    ASSERT_EQ(out->Info().base_margin_.Size(), kRows * kClasses);

    for (auto const &in_batch : p_m->GetBatches<SparsePage>()) {
      auto in_page = in_batch.GetView();
      for (auto const &out_batch : out->GetBatches<SparsePage>()) {
        auto out_page = out_batch.GetView();
        for (size_t i = 0; i < kRows; ++i) {
          auto out_inst = out_page[i];
          auto in_inst = in_page[i];
          ASSERT_EQ(out_inst.size() * 2, in_inst.size()) << i;
          for (size_t j = 0; j < kSliceSize; ++j) {
            auto const slice_start = kSliceSize * slice;
            ASSERT_EQ(in_inst[slice_start + j].fvalue, out_inst[j].fvalue);
            ASSERT_EQ(in_inst[slice_start + j].index, out_inst[j].index);
          }

          ASSERT_EQ(p_m->Info().labels_lower_bound_.HostVector().at(i),
                    out->Info().labels_lower_bound_.HostVector().at(i));
          ASSERT_EQ(p_m->Info().labels_upper_bound_.HostVector().at(i),
                    out->Info().labels_upper_bound_.HostVector().at(i));
          ASSERT_EQ(p_m->Info().weights_.HostVector().at(i), out->Info().weights_.HostVector().at(i));

          auto out_margin = out->Info().base_margin_.View(Context::kCpuId);
          auto in_margin = margin.View(Context::kCpuId);
          for (size_t j = 0; j < kClasses; ++j) {
            ASSERT_EQ(out_margin(i, j), in_margin(i, j));
          }
        }
      }
    }

    ASSERT_EQ(out->Info().num_col_, out->Info().num_col_);
    ASSERT_EQ(out->Info().num_row_, kRows);
    ASSERT_EQ(out->Info().num_nonzero_, kRows * kSliceSize);  // dense
    ASSERT_EQ(out->Info().data_split_mode, DataSplitMode::kCol);
  }
}

TEST(SimpleDMatrix, SaveLoadBinary) {
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/simple.libsvm";
  CreateSimpleTestData(tmp_file);
  xgboost::DMatrix * dmat = xgboost::DMatrix::Load(UriSVM(tmp_file));
  data::SimpleDMatrix *simple_dmat = dynamic_cast<data::SimpleDMatrix*>(dmat);

  const std::string tmp_binfile = tempdir.path + "/csr_source.binary";
  simple_dmat->SaveToLocalFile(tmp_binfile);
  xgboost::DMatrix * dmat_read = xgboost::DMatrix::Load(tmp_binfile);

  EXPECT_EQ(dmat->Info().num_col_, dmat_read->Info().num_col_);
  EXPECT_EQ(dmat->Info().num_row_, dmat_read->Info().num_row_);
  EXPECT_EQ(dmat->Info().num_row_, dmat_read->Info().num_row_);

  // Test we have non-empty batch
  EXPECT_EQ(dmat->GetBatches<xgboost::SparsePage>().begin().AtEnd(), false);

  auto row_iter = dmat->GetBatches<xgboost::SparsePage>().begin();
  auto row_iter_read = dmat_read->GetBatches<xgboost::SparsePage>().begin();
  // Test the data read into the first row
  auto first_row = (*row_iter).GetView()[0];
  auto first_row_read = (*row_iter_read).GetView()[0];
  EXPECT_EQ(first_row.size(), first_row_read.size());
  EXPECT_EQ(first_row[2].index, first_row_read[2].index);
  EXPECT_EQ(first_row[2].fvalue, first_row_read[2].fvalue);
  delete dmat;
  delete dmat_read;
}

TEST(SimpleDMatrix, Threads) {
  size_t constexpr kRows{16};
  size_t constexpr kCols{8};
  HostDeviceVector<float> data;
  auto arr_str = RandomDataGenerator{kRows, kCols, 0.0}.GenerateArrayInterface(&data);
  auto adapter = data::ArrayAdapter{StringView{arr_str}};
  std::unique_ptr<DMatrix> p_fmat{
      DMatrix::Create(&adapter, std::numeric_limits<float>::quiet_NaN(), 0, "")};
  ASSERT_EQ(p_fmat->Ctx()->Threads(), AllThreadsForTest());
}
