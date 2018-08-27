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
  EXPECT_EQ(dmat->Info().labels_.Size(), dmat->Info().num_row_);

  delete dmat;
}

TEST(SimpleDMatrix, RowAccess) {
  std::string tmp_file = CreateSimpleTestData();
  xgboost::DMatrix * dmat = xgboost::DMatrix::Load(tmp_file, false, false);
  std::remove(tmp_file.c_str());

  // Loop over the batches and count the records
  long row_count = 0;
  for(auto &batch:dmat->GetRowBatches())
  {
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

TEST(SimpleDMatrix, ColAccessWithoutBatches) {
  std::string tmp_file = CreateSimpleTestData();
  xgboost::DMatrix * dmat = xgboost::DMatrix::Load(tmp_file, true, false);
  std::remove(tmp_file.c_str());

  // Sorted column access
  EXPECT_EQ(dmat->GetColDensity(0), 1);
  EXPECT_EQ(dmat->GetColDensity(1), 0.5);
  ASSERT_TRUE(dmat->SingleColBlock());

  // Loop over the batches and assert the data is as expected
  long num_col_batch = 0;
  for (const auto &batch : dmat->GetSortedColumnBatches()) {
    num_col_batch += 1;
    EXPECT_EQ(batch.Size(), dmat->Info().num_col_)
        << "Expected batch size = number of cells as #batches is 1.";
  }
  EXPECT_EQ(num_col_batch, 1) << "Expected number of batches to be 1";
}
