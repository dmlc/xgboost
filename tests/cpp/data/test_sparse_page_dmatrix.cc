// Copyright by Contributors
#include <xgboost/data.h>
#include "../../../src/data/sparse_page_dmatrix.h"

#include "../helpers.h"

TEST(SparsePageDMatrix, MetaInfo) {
  std::string tmp_file = CreateSimpleTestData();
  xgboost::DMatrix * dmat = xgboost::DMatrix::Load(
    tmp_file + "#" + tmp_file + ".cache", false, false);
  std::remove(tmp_file.c_str());
  std::cout << tmp_file << std::endl;
  EXPECT_TRUE(FileExists(tmp_file + ".cache"));

  // Test the metadata that was parsed
  EXPECT_EQ(dmat->Info().num_row_, 2);
  EXPECT_EQ(dmat->Info().num_col_, 5);
  EXPECT_EQ(dmat->Info().num_nonzero_, 6);
  EXPECT_EQ(dmat->Info().labels_.size(), dmat->Info().num_row_);

  // Clean up of external memory files
  std::remove((tmp_file + ".cache").c_str());
  std::remove((tmp_file + ".cache.row.page").c_str());
}

TEST(SparsePageDMatrix, RowAccess) {
  std::string tmp_file = CreateSimpleTestData();
  xgboost::DMatrix * dmat = xgboost::DMatrix::Load(
    tmp_file + "#" + tmp_file + ".cache", true, false);
  std::remove(tmp_file.c_str());
  EXPECT_TRUE(FileExists(tmp_file + ".cache.row.page"));

  auto row_iter = dmat->RowIterator();
  // Loop over the batches and count the records
  long row_count = 0;
  row_iter->BeforeFirst();
  while (row_iter->Next()) row_count += row_iter->Value().Size();
  EXPECT_EQ(row_count, dmat->Info().num_row_);
  // Test the data read into the first row
  row_iter->BeforeFirst();
  row_iter->Next();
  auto first_row = row_iter->Value()[0];
  ASSERT_EQ(first_row.length, 3);
  EXPECT_EQ(first_row[2].index, 2);
  EXPECT_EQ(first_row[2].fvalue, 20);
  row_iter = nullptr;

  // Clean up of external memory files
  std::remove((tmp_file + ".cache").c_str());
  std::remove((tmp_file + ".cache.row.page").c_str());
}

TEST(SparsePageDMatrix, ColAcess) {
  std::string tmp_file = CreateSimpleTestData();
  xgboost::DMatrix * dmat = xgboost::DMatrix::Load(
    tmp_file + "#" + tmp_file + ".cache", true, false);
  std::remove(tmp_file.c_str());
  EXPECT_FALSE(FileExists(tmp_file + ".cache.col.page"));

  EXPECT_EQ(dmat->HaveColAccess(true), false);
  const std::vector<bool> enable(dmat->Info().num_col_, true);
  dmat->InitColAccess(1, true); // Max 1 row per patch
  ASSERT_EQ(dmat->HaveColAccess(true), true);
  EXPECT_TRUE(FileExists(tmp_file + ".cache.col.page"));

  EXPECT_EQ(dmat->GetColSize(0), 2);
  EXPECT_EQ(dmat->GetColSize(1), 1);
  EXPECT_EQ(dmat->GetColDensity(0), 1);
  EXPECT_EQ(dmat->GetColDensity(1), 0.5);

  auto col_iter = dmat->ColIterator();
  // Loop over the batches and assert the data is as expected
  long num_col_batch = 0;
  col_iter->BeforeFirst();
  while (col_iter->Next()) {
    num_col_batch += 1;
    EXPECT_EQ(col_iter->Value().Size(), dmat->Info().num_col_)
      << "Expected batch size to be same as num_cols as max_row_perbatch is 1.";
  }
  EXPECT_EQ(num_col_batch, dmat->Info().num_row_)
    << "Expected num batches to be same as num_rows as max_row_perbatch is 1";
  col_iter = nullptr;

  std::remove((tmp_file + ".cache").c_str());
  std::remove((tmp_file + ".cache.col.page").c_str());
  std::remove((tmp_file + ".cache.row.page").c_str());
}
