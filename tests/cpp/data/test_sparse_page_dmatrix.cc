// Copyright by Contributors
#include <xgboost/data.h>
#include <dmlc/filesystem.h>
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
  // Create sufficiently large data to make two row pages
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/big.libsvm";
  CreateBigTestData(tmp_file, 5000000);
  xgboost::DMatrix * dmat = xgboost::DMatrix::Load(
    tmp_file + "#" + tmp_file + ".cache", true, false);
  EXPECT_TRUE(FileExists(tmp_file + ".cache.row.page"));

  // Loop over the batches and count the records
  long row_count = 0;
  for (auto &batch : dmat->GetRowBatches()) {
    row_count += batch.Size();
  }
  EXPECT_EQ(row_count, dmat->Info().num_row_);

  // Test the data read into the first row
  auto &batch = *dmat->GetRowBatches().begin();
  auto first_row = batch[0];
  ASSERT_EQ(first_row.size(), 3);
  EXPECT_EQ(first_row[2].index, 2);
  EXPECT_EQ(first_row[2].fvalue, 20);

  delete dmat;
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
  for (auto col_batch : dmat->GetSortedColumnBatches()) {
    EXPECT_EQ(col_batch.Size(), dmat->Info().num_col_);
    EXPECT_EQ(col_batch[1][0].fvalue, 10.0f);
    EXPECT_EQ(col_batch[1].size(), 1);
  }

  // Loop over the batches and assert the data is as expected
  for (auto col_batch : dmat->GetColumnBatches()) {
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
