// Copyright by Contributors
#include <dmlc/filesystem.h>
#include <xgboost/data.h>
#include "../../../src/data/simple_dmatrix.h"

#include "../../../src/data/adapter.h"
#include "../helpers.h"

using namespace xgboost;  // NOLINT

TEST(SimpleDMatrix, MetaInfo) {
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/simple.libsvm";
  CreateSimpleTestData(tmp_file);
  xgboost::DMatrix *dmat = xgboost::DMatrix::Load(tmp_file, true, false);

  // Test the metadata that was parsed
  EXPECT_EQ(dmat->Info().num_row_, 2);
  EXPECT_EQ(dmat->Info().num_col_, 5);
  EXPECT_EQ(dmat->Info().num_nonzero_, 6);
  EXPECT_EQ(dmat->Info().labels_.Size(), dmat->Info().num_row_);

  delete dmat;
}

TEST(SimpleDMatrix, RowAccess) {
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/simple.libsvm";
  CreateSimpleTestData(tmp_file);
  xgboost::DMatrix *dmat = xgboost::DMatrix::Load(tmp_file, false, false);

  // Loop over the batches and count the records
  int64_t row_count = 0;
  for (auto &batch : dmat->GetBatches<xgboost::SparsePage>()) {
    row_count += batch.Size();
  }
  EXPECT_EQ(row_count, dmat->Info().num_row_);
  // Test the data read into the first row
  auto &batch = *dmat->GetBatches<xgboost::SparsePage>().begin();
  auto first_row = batch[0];
  ASSERT_EQ(first_row.size(), 3);
  EXPECT_EQ(first_row[2].index, 2);
  EXPECT_EQ(first_row[2].fvalue, 20);

  delete dmat;
}

TEST(SimpleDMatrix, ColAccessWithoutBatches) {
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/simple.libsvm";
  CreateSimpleTestData(tmp_file);
  xgboost::DMatrix *dmat = xgboost::DMatrix::Load(tmp_file, true, false);

  // Loop over the batches and assert the data is as expected
  int64_t num_col_batch = 0;
  for (const auto &batch : dmat->GetBatches<xgboost::SortedCSCPage>()) {
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
  data::SimpleDMatrix dmat(&csr_adapter,
                           std::numeric_limits<float>::quiet_NaN(), 1);
  CHECK_EQ(dmat.Info().num_nonzero_, 0);
  CHECK_EQ(dmat.Info().num_row_, 0);
  CHECK_EQ(dmat.Info().num_col_, 0);
  for (auto &batch : dmat.GetBatches<SparsePage>()) {
    CHECK_EQ(batch.Size(), 0);
  }

  data::DenseAdapter dense_adapter(nullptr, 0, 0);
  dmat = data::SimpleDMatrix(&dense_adapter,
                             std::numeric_limits<float>::quiet_NaN(), 1);
  CHECK_EQ(dmat.Info().num_nonzero_, 0);
  CHECK_EQ(dmat.Info().num_row_, 0);
  CHECK_EQ(dmat.Info().num_col_, 0);
  for (auto &batch : dmat.GetBatches<SparsePage>()) {
    CHECK_EQ(batch.Size(), 0);
  }

  data::CSCAdapter csc_adapter(nullptr, nullptr, nullptr, 0, 0);
  dmat = data::SimpleDMatrix(&csc_adapter,
                             std::numeric_limits<float>::quiet_NaN(), 1);
  CHECK_EQ(dmat.Info().num_nonzero_, 0);
  CHECK_EQ(dmat.Info().num_row_, 0);
  CHECK_EQ(dmat.Info().num_col_, 0);
  for (auto &batch : dmat.GetBatches<SparsePage>()) {
    CHECK_EQ(batch.Size(), 0);
  }
}

TEST(SimpleDMatrix, MissingData) {
  std::vector<float> data{0.0, std::nanf(""), 1.0};
  std::vector<unsigned> feature_idx = {0, 1, 0};
  std::vector<size_t> row_ptr = {0, 2, 3};

  data::CSRAdapter adapter(row_ptr.data(), feature_idx.data(), data.data(), 2,
                           3, 2);
  data::SimpleDMatrix dmat(&adapter, std::numeric_limits<float>::quiet_NaN(),
                           1);
  CHECK_EQ(dmat.Info().num_nonzero_, 2);
  dmat = data::SimpleDMatrix(&adapter, 1.0, 1);
  CHECK_EQ(dmat.Info().num_nonzero_, 1);
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
    for (auto i = 0ull; i < batch.Size(); i++) {
      auto inst = batch[i];
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
  auto inst = batch[0];
  EXPECT_EQ(inst[0].fvalue, 1);
  EXPECT_EQ(inst[0].index, 0);
  EXPECT_EQ(inst[1].fvalue, 2);
  EXPECT_EQ(inst[1].index, 1);

  inst = batch[1];
  EXPECT_EQ(inst[0].fvalue, 3);
  EXPECT_EQ(inst[0].index, 0);
  EXPECT_EQ(inst[1].fvalue, 4);
  EXPECT_EQ(inst[1].index, 1);

  inst = batch[2];
  EXPECT_EQ(inst[0].fvalue, 5);
  EXPECT_EQ(inst[0].index, 1);
}

TEST(SimpleDMatrix, FromFile) {
  std::string filename = "test.libsvm";
  CreateBigTestData(filename, 3 * 5);
  std::unique_ptr<dmlc::Parser<uint32_t>> parser(
      dmlc::Parser<uint32_t>::Create(filename.c_str(), 0, 1, "auto"));
  data::FileAdapter adapter(parser.get());
  data::SimpleDMatrix dmat(&adapter, std::numeric_limits<float>::quiet_NaN(),
                           1);
  for (auto &batch : dmat.GetBatches<SparsePage>()) {
    EXPECT_EQ(batch.Size(), 5);
    EXPECT_EQ(batch.offset.HostVector(),
              std::vector<bst_row_t>({0, 3, 6, 9, 12, 15}));
    EXPECT_EQ(batch.base_rowid, 0);

    for (auto i = 0ull; i < batch.Size(); i++) {
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
  }
}

TEST(SimpleDMatrix, Slice) {
  const int kRows = 6;
  const int kCols = 2;
  auto pp_dmat = CreateDMatrix(kRows, kCols, 1.0);
  auto p_dmat = *pp_dmat;
  auto &labels = p_dmat->Info().labels_.HostVector();
  auto &weights = p_dmat->Info().weights_.HostVector();
  auto &base_margin = p_dmat->Info().base_margin_.HostVector();
  weights.resize(kRows);
  labels.resize(kRows);
  base_margin.resize(kRows);
  std::iota(labels.begin(), labels.end(), 0);
  std::iota(weights.begin(), weights.end(), 0);
  std::iota(base_margin.begin(), base_margin.end(), 0);

  std::vector<int> ridx_set = {1, 3, 5};
  data::DMatrixSliceAdapter adapter(p_dmat.get(),
                                    {ridx_set.data(), ridx_set.size()});
  EXPECT_EQ(adapter.NumRows(), ridx_set.size());
  data::SimpleDMatrix new_dmat(&adapter,
                               std::numeric_limits<float>::quiet_NaN(), 1);

  EXPECT_EQ(new_dmat.Info().num_row_, ridx_set.size());

  auto &old_batch = *p_dmat->GetBatches<SparsePage>().begin();
  auto &new_batch = *new_dmat.GetBatches<SparsePage>().begin();
  for (auto i = 0ull; i < ridx_set.size(); i++) {
    EXPECT_EQ(new_dmat.Info().labels_.HostVector()[i],
              p_dmat->Info().labels_.HostVector()[ridx_set[i]]);
    EXPECT_EQ(new_dmat.Info().weights_.HostVector()[i],
              p_dmat->Info().weights_.HostVector()[ridx_set[i]]);
    EXPECT_EQ(new_dmat.Info().base_margin_.HostVector()[i],
              p_dmat->Info().base_margin_.HostVector()[ridx_set[i]]);
    auto old_inst = old_batch[ridx_set[i]];
    auto new_inst = new_batch[i];
    ASSERT_EQ(old_inst.size(), new_inst.size());
    for (auto j = 0ull; j < old_inst.size(); j++) {
      EXPECT_EQ(old_inst[j], new_inst[j]);
    }
  }

  delete pp_dmat;
};

TEST(SimpleDMatrix, SaveLoadBinary) {
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/simple.libsvm";
  CreateSimpleTestData(tmp_file);
  xgboost::DMatrix * dmat = xgboost::DMatrix::Load(tmp_file, true, false);
  data::SimpleDMatrix *simple_dmat = dynamic_cast<data::SimpleDMatrix*>(dmat);

  const std::string tmp_binfile = tempdir.path + "/csr_source.binary";
  simple_dmat->SaveToLocalFile(tmp_binfile);
  xgboost::DMatrix * dmat_read = xgboost::DMatrix::Load(tmp_binfile, true, false);

  EXPECT_EQ(dmat->Info().num_col_, dmat_read->Info().num_col_);
  EXPECT_EQ(dmat->Info().num_row_, dmat_read->Info().num_row_);
  EXPECT_EQ(dmat->Info().num_row_, dmat_read->Info().num_row_);

  // Test we have non-empty batch
  EXPECT_EQ(dmat->GetBatches<xgboost::SparsePage>().begin().AtEnd(), false);

  auto row_iter = dmat->GetBatches<xgboost::SparsePage>().begin();
  auto row_iter_read = dmat_read->GetBatches<xgboost::SparsePage>().begin();
  // Test the data read into the first row
  auto first_row = (*row_iter)[0];
  auto first_row_read = (*row_iter_read)[0];
  EXPECT_EQ(first_row.size(), first_row_read.size());
  EXPECT_EQ(first_row[2].index, first_row_read[2].index);
  EXPECT_EQ(first_row[2].fvalue, first_row_read[2].fvalue);
  delete dmat;
  delete dmat_read;
}
