// Copyright by Contributors
#include <dmlc/filesystem.h>
#include <xgboost/data.h>
#include "../../../src/data/sparse_page_dmatrix.h"
#include "../../../src/data/adapter.h"
#include "../helpers.h"
#include <gtest/gtest.h>

using namespace xgboost;  // NOLINT

TEST(SparsePageDMatrix, MetaInfo) {
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/simple.libsvm";
  CreateSimpleTestData(tmp_file);
  xgboost::DMatrix * dmat = xgboost::DMatrix::Load(
    tmp_file + "#" + tmp_file + ".cache", false, false);
  std::cout << tmp_file << std::endl;
  EXPECT_TRUE(FileExists(tmp_file + ".cache"));

  // Test the metadata that was parsed
  EXPECT_EQ(dmat->Info().num_row_, 2);
  EXPECT_EQ(dmat->Info().num_col_, 5);
  EXPECT_EQ(dmat->Info().num_nonzero_, 6);
  EXPECT_EQ(dmat->Info().labels_.Size(), dmat->Info().num_row_);

  delete dmat;
}

TEST(SparsePageDMatrix, RowAccess) {
  dmlc::TemporaryDirectory tmpdir;
  std::string filename = tmpdir.path + "/big.libsvm";
  std::unique_ptr<xgboost::DMatrix> dmat =
      xgboost::CreateSparsePageDMatrix(12, 64, filename);

  // Test the data read into the first row
  auto &batch = *dmat->GetBatches<xgboost::SparsePage>().begin();
  auto first_row = batch[0];
  ASSERT_EQ(first_row.size(), 3);
  EXPECT_EQ(first_row[2].index, 2);
  EXPECT_EQ(first_row[2].fvalue, 20);
}

TEST(SparsePageDMatrix, ColAccess) {
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/simple.libsvm";
  CreateSimpleTestData(tmp_file);
  xgboost::DMatrix * dmat = xgboost::DMatrix::Load(
    tmp_file + "#" + tmp_file + ".cache", true, false);

  EXPECT_EQ(dmat->GetColDensity(0), 1);
  EXPECT_EQ(dmat->GetColDensity(1), 0.5);

  // Loop over the batches and assert the data is as expected
  for (auto col_batch : dmat->GetBatches<xgboost::SortedCSCPage>()) {
    EXPECT_EQ(col_batch.Size(), dmat->Info().num_col_);
    EXPECT_EQ(col_batch[1][0].fvalue, 10.0f);
    EXPECT_EQ(col_batch[1].size(), 1);
  }

  // Loop over the batches and assert the data is as expected
  for (auto col_batch : dmat->GetBatches<xgboost::CSCPage>()) {
    EXPECT_EQ(col_batch.Size(), dmat->Info().num_col_);
    EXPECT_EQ(col_batch[1][0].fvalue, 10.0f);
    EXPECT_EQ(col_batch[1].size(), 1);
  }

  EXPECT_TRUE(FileExists(tmp_file + ".cache"));
  EXPECT_TRUE(FileExists(tmp_file + ".cache.row.page"));
  EXPECT_TRUE(FileExists(tmp_file + ".cache.col.page"));
  EXPECT_TRUE(FileExists(tmp_file + ".cache.sorted.col.page"));

  delete dmat;
}

// Multi-batches access
TEST(SparsePageDMatrix, ColAccessBatches) {
  dmlc::TemporaryDirectory tmpdir;
  std::string filename = tmpdir.path + "/big.libsvm";
  // Create multiple sparse pages
  std::unique_ptr<xgboost::DMatrix> dmat {
    xgboost::CreateSparsePageDMatrix(1024, 1024, filename)
  };
  auto n_threads = omp_get_max_threads();
  omp_set_num_threads(16);
  for (auto const& page : dmat->GetBatches<xgboost::CSCPage>()) {
    ASSERT_EQ(dmat->Info().num_col_, page.Size());
  }
  omp_set_num_threads(n_threads);
}


TEST(SparsePageDMatrix, Empty) {
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/simple.libsvm";
  std::vector<float> data{};
  std::vector<unsigned> feature_idx = {};
  std::vector<size_t> row_ptr = {};

  data::CSRAdapter csr_adapter(row_ptr.data(), feature_idx.data(), data.data(), 0, 0, 0);
  data::SparsePageDMatrix dmat(&csr_adapter,
    std::numeric_limits<float>::quiet_NaN(), 1,tmp_file);
  EXPECT_EQ(dmat.Info().num_nonzero_, 0);
  EXPECT_EQ(dmat.Info().num_row_, 0);
  EXPECT_EQ(dmat.Info().num_col_, 0);
  for (auto &batch : dmat.GetBatches<SparsePage>()) {
    EXPECT_EQ(batch.Size(), 0);
  }

  data::DenseAdapter dense_adapter(nullptr, 0, 0, 0);
   data::SparsePageDMatrix dmat2(&dense_adapter,
    std::numeric_limits<float>::quiet_NaN(), 1,tmp_file);
  EXPECT_EQ(dmat2.Info().num_nonzero_, 0);
  EXPECT_EQ(dmat2.Info().num_row_, 0);
  EXPECT_EQ(dmat2.Info().num_col_, 0);
  for (auto &batch : dmat2.GetBatches<SparsePage>()) {
    EXPECT_EQ(batch.Size(), 0);
  }

  data::CSCAdapter csc_adapter(nullptr, nullptr, nullptr, 0, 0);
  data::SparsePageDMatrix dmat3(&csc_adapter,
    std::numeric_limits<float>::quiet_NaN(), 1,tmp_file);
  EXPECT_EQ(dmat3.Info().num_nonzero_, 0);
  EXPECT_EQ(dmat3.Info().num_row_, 0);
  EXPECT_EQ(dmat3.Info().num_col_, 0);
  for (auto &batch : dmat3.GetBatches<SparsePage>()) {
    EXPECT_EQ(batch.Size(), 0);
  }
}

TEST(SparsePageDMatrix, MissingData) {
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/simple.libsvm";
  std::vector<float> data{0.0, std::nanf(""), 1.0};
  std::vector<unsigned> feature_idx = {0, 1, 0};
  std::vector<size_t> row_ptr = {0, 2, 3};

  data::CSRAdapter adapter(row_ptr.data(), feature_idx.data(), data.data(), 2, 3, 2);
  data::SparsePageDMatrix dmat(&adapter, std::numeric_limits<float>::quiet_NaN(), 1,tmp_file);
  EXPECT_EQ(dmat.Info().num_nonzero_, 2);

  const std::string tmp_file2 = tempdir.path + "/simple2.libsvm";
  data::SparsePageDMatrix dmat2(&adapter, 1.0, 1,tmp_file2);
  EXPECT_EQ(dmat2.Info().num_nonzero_, 1);
}

TEST(SparsePageDMatrix, EmptyRow) {
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/simple.libsvm";
  std::vector<float> data{0.0, 1.0};
  std::vector<unsigned> feature_idx = {0, 1};
  std::vector<size_t> row_ptr = {0, 2, 2};

  data::CSRAdapter adapter(row_ptr.data(), feature_idx.data(), data.data(), 2, 2, 2);
  data::SparsePageDMatrix dmat(&adapter, std::numeric_limits<float>::quiet_NaN(), 1,tmp_file);
  EXPECT_EQ(dmat.Info().num_nonzero_, 2);
  EXPECT_EQ(dmat.Info().num_row_, 2);
  EXPECT_EQ(dmat.Info().num_col_, 2);
}

TEST(SparsePageDMatrix, FromDense) {
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/simple.libsvm";
  int m = 3;
  int n = 2;
  std::vector<float> data = {1, 2, 3, 4, 5, 6};
  data::DenseAdapter adapter(data.data(), m, m * n, n);
  data::SparsePageDMatrix dmat(
      &adapter, std::numeric_limits<float>::quiet_NaN(), 1, tmp_file);
  EXPECT_EQ(dmat.Info().num_col_, 2);
  EXPECT_EQ(dmat.Info().num_row_, 3);
  EXPECT_EQ(dmat.Info().num_nonzero_, 6);

  for (auto &batch : dmat.GetBatches<SparsePage>()) {
    for (auto i = 0ull; i < batch.Size(); i++) {
      auto inst = batch[i];
      for(auto j = 0ull; j < inst.size(); j++)
      {
        EXPECT_EQ(inst[j].fvalue, data[i*n+j]);
        EXPECT_EQ(inst[j].index, j);
      }
    }
  }
}

TEST(SparsePageDMatrix, FromCSC) {
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/simple.libsvm";
  std::vector<float> data = {1, 3, 2, 4, 5};
  std::vector<unsigned> row_idx = {0, 1, 0, 1, 2};
  std::vector<size_t> col_ptr = {0, 2, 5};
  data::CSCAdapter adapter(col_ptr.data(), row_idx.data(), data.data(), 2, 3);
  data::SparsePageDMatrix dmat(
      &adapter, std::numeric_limits<float>::quiet_NaN(), -1, tmp_file);
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

TEST(SparsePageDMatrix, FromFile) {
  std::string filename = "test.libsvm";
  CreateBigTestData(filename,20);
  std::unique_ptr<dmlc::Parser<uint32_t>> parser(
    dmlc::Parser<uint32_t>::Create(filename.c_str(), 0, 1, "auto"));
  data::FileAdapter adapter(parser.get());
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/simple.libsvm";
  data::SparsePageDMatrix dmat(
      &adapter, std::numeric_limits<float>::quiet_NaN(), -1, tmp_file, 1);
  
  for (auto &batch : dmat.GetBatches<SparsePage>()) {
    std::vector<bst_row_t> expected_offset(batch.Size() + 1);
    int n = -3;
    std::generate(expected_offset.begin(), expected_offset.end(),
                  [&n] { return n += 3; });
    EXPECT_EQ(batch.offset.HostVector(), expected_offset);

    if (batch.base_rowid % 2 == 0) {
      EXPECT_EQ(batch[0][0].index, 0);
      EXPECT_EQ(batch[0][1].index, 1);
      EXPECT_EQ(batch[0][2].index, 2);
    } else {
      EXPECT_EQ(batch[0][0].index, 0);
      EXPECT_EQ(batch[0][1].index, 3);
      EXPECT_EQ(batch[0][2].index, 4);
    }
  }
}
