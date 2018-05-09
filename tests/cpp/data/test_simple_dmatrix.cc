// Copyright by Contributors
#include <xgboost/data.h>
#include "../../../src/data/simple_dmatrix.h"

#include "../helpers.h"

TEST(SimpleDMatrix, MetaInfo) {
  std::string tmp_file = CreateSimpleTestData();
  xgboost::DMatrix * dmat = xgboost::DMatrix::Load(tmp_file, true, false);
  std::remove(tmp_file.c_str());

  // Test the metadata that was parsed
  EXPECT_EQ(dmat->Info().num_row_, 2);
  EXPECT_EQ(dmat->Info().num_col_, 5);
  EXPECT_EQ(dmat->Info().num_nonzero_, 6);
  EXPECT_EQ(dmat->Info().labels_.size(), dmat->Info().num_row_);
}

TEST(SimpleDMatrix, RowAccess) {
  std::string tmp_file = CreateSimpleTestData();
  xgboost::DMatrix * dmat = xgboost::DMatrix::Load(tmp_file, false, false);
  std::remove(tmp_file.c_str());

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
}

TEST(SimpleDMatrix, ColAccessWithoutBatches) {
  std::string tmp_file = CreateSimpleTestData();
  xgboost::DMatrix * dmat = xgboost::DMatrix::Load(tmp_file, true, false);
  std::remove(tmp_file.c_str());

  // Unsorted column access
  const std::vector<bool> enable(dmat->Info().num_col_, true);
  EXPECT_EQ(dmat->HaveColAccess(false), false);
  dmat->InitColAccess(dmat->Info().num_row_, false);
  dmat->InitColAccess(0, false); // Calling it again should not change it
  ASSERT_EQ(dmat->HaveColAccess(false), true);

  // Sorted column access
  EXPECT_EQ(dmat->HaveColAccess(true), false);
  dmat->InitColAccess(dmat->Info().num_row_, true);
  dmat->InitColAccess(0, true); // Calling it again should not change it
  ASSERT_EQ(dmat->HaveColAccess(true), true);

  EXPECT_EQ(dmat->GetColSize(0), 2);
  EXPECT_EQ(dmat->GetColSize(1), 1);
  EXPECT_EQ(dmat->GetColDensity(0), 1);
  EXPECT_EQ(dmat->GetColDensity(1), 0.5);
  ASSERT_TRUE(dmat->SingleColBlock());

  auto* col_iter = dmat->ColIterator();
  // Loop over the batches and assert the data is as expected
  long num_col_batch = 0;
  col_iter->BeforeFirst();
  while (col_iter->Next()) {
    num_col_batch += 1;
    EXPECT_EQ(col_iter->Value().Size(), dmat->Info().num_col_)
      << "Expected batch size = number of cells as #batches is 1.";
    for (int i = 0; i < static_cast<int>(col_iter->Value().Size()); ++i) {
      EXPECT_EQ(col_iter->Value()[i].length, dmat->GetColSize(i))
        << "Expected length of each colbatch = colsize as #batches is 1.";
    }
  }
  EXPECT_EQ(num_col_batch, 1) << "Expected number of batches to be 1";
  col_iter = nullptr;
}
