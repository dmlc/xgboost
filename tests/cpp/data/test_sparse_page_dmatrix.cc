// Copyright by Contributors
#include <xgboost/data.h>
#include "../../../src/data/sparse_page_dmatrix.h"

#include "../helpers.h"

TEST(SparsePageDMatrix, MetaInfo) {
  std::string tmp_file = CreateSimpleTestData();
  xgboost::DMatrix * dmat = xgboost::DMatrix::Load(
    tmp_file + "#" + tmp_file + ".cache", false, false);
  std::cout << tmp_file << std::endl;
  EXPECT_TRUE(FileExists(tmp_file + ".cache"));

  // Test the metadata that was parsed
  EXPECT_EQ(dmat->Info().num_row_, 2);
  EXPECT_EQ(dmat->Info().num_col_, 5);
  EXPECT_EQ(dmat->Info().num_nonzero_, 6);
  EXPECT_EQ(dmat->Info().labels_.Size(), dmat->Info().num_row_);

  // Clean up of external memory files
  std::remove(tmp_file.c_str());
  std::remove((tmp_file + ".cache").c_str());
  std::remove((tmp_file + ".cache.row.page").c_str());

  delete dmat;
}

TEST(SparsePageDMatrix, RowAccess) {
  // Create sufficiently large data to make two row pages
  std::string tmp_file = CreateBigTestData(5000000);
  xgboost::DMatrix * dmat = xgboost::DMatrix::Load(
    tmp_file + "#" + tmp_file + ".cache", true, false);
  std::remove(tmp_file.c_str());
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

  // Clean up of external memory files
  std::remove((tmp_file + ".cache").c_str());
  std::remove((tmp_file + ".cache.row.page").c_str());
}

TEST(SparsePageDMatrix, ColAccess) {
  std::string tmp_file = CreateSimpleTestData();
  xgboost::DMatrix * dmat = xgboost::DMatrix::Load(
    tmp_file + "#" + tmp_file + ".cache", true, false);
  std::remove(tmp_file.c_str());

  EXPECT_EQ(dmat->GetColDensity(0), 1);
  EXPECT_EQ(dmat->GetColDensity(1), 0.5);

  // Loop over the batches and assert the data is as expected
  for (auto col_batch : dmat->GetSortedColumnBatches()) {
    EXPECT_EQ(col_batch.Size(), dmat->Info().num_col_);
    EXPECT_EQ(col_batch[1][0].fvalue, 10.0f);
    EXPECT_EQ(col_batch[1].size(), 1);
  }
  EXPECT_EQ(num_col_batch, dmat->Info().num_row_)
    << "Expected num batches to be same as num_rows as max_row_perbatch is 1";
  col_iter = nullptr;
  delete dmat;

  std::remove((tmp_file + ".cache").c_str());
  std::remove((tmp_file + ".cache.row.page").c_str());
  std::remove((tmp_file + ".cache.col.meta").c_str());
}
