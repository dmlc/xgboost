// Copyright by Contributors
#include <xgboost/data.h>
#include "../../../src/data/simple_csr_source.h"

#include "../helpers.h"

TEST(SimpleCSRSource, SaveLoadBinary) {
  std::string tmp_file = CreateSimpleTestData();
  xgboost::DMatrix * dmat = xgboost::DMatrix::Load(tmp_file, true, false);
  std::remove(tmp_file.c_str());

  std::string tmp_binfile = TempFileName();
  dmat->SaveToLocalFile(tmp_binfile);
  xgboost::DMatrix * dmat_read = xgboost::DMatrix::Load(tmp_binfile, true, false);
  std::remove(tmp_binfile.c_str());

  EXPECT_EQ(dmat->Info().num_col_, dmat_read->Info().num_col_);
  EXPECT_EQ(dmat->Info().num_row_, dmat_read->Info().num_row_);
  EXPECT_EQ(dmat->Info().num_row_, dmat_read->Info().num_row_);

  // Test we have non-empty batch
  EXPECT_EQ(dmat->GetRowBatches().begin().AtEnd(), false);

  auto row_iter = dmat->GetRowBatches().begin();
  auto row_iter_read = dmat_read->GetRowBatches().begin();
  // Test the data read into the first row
  auto first_row = (*row_iter)[0];
  auto first_row_read = (*row_iter_read)[0];
  EXPECT_EQ(first_row.size(), first_row_read.size());
  EXPECT_EQ(first_row[2].index, first_row_read[2].index);
  EXPECT_EQ(first_row[2].fvalue, first_row_read[2].fvalue);
  delete dmat;
  delete dmat_read;
}
