// Copyright by Contributors
#include <xgboost/data.h>
#include "../../../src/data/simple_dmatrix.h"

#include "../helpers.h"

TEST(SimpleDMatrix, MetaInfo) {
  std::string tmp_file = CreateSimpleTestData();
  xgboost::DMatrix * dmat = xgboost::DMatrix::Load(tmp_file, true, false);
  std::remove(tmp_file.c_str());

  // Test the metadata that was parsed
  EXPECT_EQ(dmat->info().num_row, 2);
  EXPECT_EQ(dmat->info().num_col, 5);
  EXPECT_EQ(dmat->info().num_nonzero, 6);
  EXPECT_EQ(dmat->info().labels.size(), dmat->info().num_row);
}

TEST(SimpleDMatrix, RowAccess) {
  std::string tmp_file = CreateSimpleTestData();
  xgboost::DMatrix * dmat = xgboost::DMatrix::Load(tmp_file, true, false);
  std::remove(tmp_file.c_str());

  dmlc::DataIter<xgboost::RowBatch> * row_iter = dmat->RowIterator();
  // Loop over the batches and count the records
  long row_count = 0;
  row_iter->BeforeFirst();
  while (row_iter->Next()) row_count += row_iter->Value().size;
  EXPECT_EQ(row_count, dmat->info().num_row);
  // Test the data read into the first row
  row_iter->BeforeFirst();
  row_iter->Next();
  xgboost::SparseBatch::Inst first_row = row_iter->Value()[0];
  ASSERT_EQ(first_row.length, 3);
  EXPECT_EQ(first_row[2].index, 2);
  EXPECT_EQ(first_row[2].fvalue, 20);
  row_iter = nullptr;
}

TEST(SimpleDMatrix, ColAccessWithoutBatches) {
  std::string tmp_file = CreateSimpleTestData();
  xgboost::DMatrix * dmat = xgboost::DMatrix::Load(tmp_file, true, false);
  std::remove(tmp_file.c_str());

  EXPECT_EQ(dmat->HaveColAccess(), false);
  const std::vector<bool> enable(dmat->info().num_col, true);
  dmat->InitColAccess(enable, 1, dmat->info().num_row);
  dmat->InitColAccess(enable, 0, 0); // Calling it again should not change it
  ASSERT_EQ(dmat->HaveColAccess(), true);

  EXPECT_EQ(dmat->GetColSize(0), 2);
  EXPECT_EQ(dmat->GetColSize(1), 1);
  EXPECT_EQ(dmat->GetColDensity(0), 1);
  EXPECT_EQ(dmat->GetColDensity(1), 0.5);
  ASSERT_TRUE(dmat->SingleColBlock());

  dmlc::DataIter<xgboost::ColBatch> * col_iter = dmat->ColIterator();
  // Loop over the batches and assert the data is as expected
  long num_col_batch = 0;
  col_iter->BeforeFirst();
  while (col_iter->Next()) {
    num_col_batch += 1;
    EXPECT_EQ(col_iter->Value().size, dmat->info().num_col)
      << "Expected batch size = number of cells as #batches is 1.";
    for (int i = 0; i < static_cast<int>(col_iter->Value().size); ++i) {
      EXPECT_EQ(col_iter->Value()[i].length, dmat->GetColSize(i))
        << "Expected length of each colbatch = colsize as #batches is 1.";
    }
  }
  EXPECT_EQ(num_col_batch, 1) << "Expected number of batches to be 1";
  col_iter = nullptr;

  std::vector<xgboost::bst_uint> sub_feats = {4, 3};
  dmlc::DataIter<xgboost::ColBatch> * sub_col_iter = dmat->ColIterator(sub_feats);
  // Loop over the batches and assert the data is as expected
  sub_col_iter->BeforeFirst();
  while (sub_col_iter->Next()) {
    EXPECT_EQ(sub_col_iter->Value().size, sub_feats.size())
      << "Expected size of a batch = number of cells in subset as #batches is 1.";
  }
  sub_col_iter = nullptr;
}

TEST(SimpleDMatrix, ColAccessWithBatches) {
  std::string tmp_file = CreateSimpleTestData();
  xgboost::DMatrix * dmat = xgboost::DMatrix::Load(tmp_file, true, false);
  std::remove(tmp_file.c_str());

  EXPECT_EQ(dmat->HaveColAccess(), false);
  const std::vector<bool> enable(dmat->info().num_col, true);
  dmat->InitColAccess(enable, 1, 1); // Max 1 row per patch
  dmat->InitColAccess(enable, 0, 0); // Calling it again should not change it
  ASSERT_EQ(dmat->HaveColAccess(), true);

  EXPECT_EQ(dmat->GetColSize(0), 2);
  EXPECT_EQ(dmat->GetColSize(1), 1);
  EXPECT_EQ(dmat->GetColDensity(0), 1);
  EXPECT_EQ(dmat->GetColDensity(1), 0.5);
  ASSERT_FALSE(dmat->SingleColBlock());

  dmlc::DataIter<xgboost::ColBatch> * col_iter = dmat->ColIterator();
  // Loop over the batches and assert the data is as expected
  long num_col_batch = 0;
  col_iter->BeforeFirst();
  while (col_iter->Next()) {
    num_col_batch += 1;
    EXPECT_EQ(col_iter->Value().size, dmat->info().num_col)
      << "Expected batch size = num_cols as max_row_perbatch is 1.";
    for (int i = 0; i < static_cast<int>(col_iter->Value().size); ++i) {
      EXPECT_LE(col_iter->Value()[i].length, 1)
        << "Expected length of each colbatch <=1 as max_row_perbatch is 1.";
    }
  }
  EXPECT_EQ(num_col_batch, dmat->info().num_row)
    << "Expected num batches = num_rows as max_row_perbatch is 1";
  col_iter = nullptr;

  // The iterator feats should ignore any numbers larger than the num_col
  std::vector<xgboost::bst_uint> sub_feats = {
    4, 3, static_cast<unsigned int>(dmat->info().num_col + 1)};
  dmlc::DataIter<xgboost::ColBatch> * sub_col_iter = dmat->ColIterator(sub_feats);
  // Loop over the batches and assert the data is as expected
  sub_col_iter->BeforeFirst();
  while (sub_col_iter->Next()) {
    EXPECT_EQ(sub_col_iter->Value().size, sub_feats.size() - 1)
      << "Expected size of a batch = number of columns in subset "
      << "as max_row_perbatch is 1.";
  }
  sub_col_iter = nullptr;
}
