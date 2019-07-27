// Copyright by Contributors
#include <dmlc/filesystem.h>
#include <xgboost/data.h>
#include <dmlc/filesystem.h>
#include <cinttypes>

#include "../../../src/data/sparse_page_dmatrix.h"

#include "../helpers.h"

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
  auto &batch = *dmat->GetBatches<xgboost::SparsePage>(xgboost::kCSR).begin();
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
  for (auto col_batch : dmat->GetBatches<xgboost::SparsePage>(xgboost::kSortedCSC)) {
    EXPECT_EQ(col_batch.Size(), dmat->Info().num_col_);
    EXPECT_EQ(col_batch[1][0].fvalue, 10.0f);
    EXPECT_EQ(col_batch[1].size(), 1);
  }

  // Loop over the batches and assert the data is as expected
  for (auto col_batch : dmat->GetBatches<xgboost::SparsePage>(xgboost::kCSC)) {
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
  for (auto const& page : dmat->GetBatches<xgboost::SparsePage>(xgboost::kCSC)) {
    ASSERT_EQ(dmat->Info().num_col_, page.Size());
  }
  omp_set_num_threads(n_threads);
}
