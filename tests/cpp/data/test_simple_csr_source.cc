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

  auto row_iter = dmat->RowIterator();
  auto row_iter_read = dmat_read->RowIterator();
  // Test the data read into the first row
  row_iter->BeforeFirst(); row_iter->Next();
  row_iter_read->BeforeFirst(); row_iter_read->Next();
  auto first_row = row_iter->Value()[0];
  auto first_row_read = row_iter_read->Value()[0];
  EXPECT_EQ(first_row.length, first_row_read.length);
  EXPECT_EQ(first_row[2].index, first_row_read[2].index);
  EXPECT_EQ(first_row[2].fvalue, first_row_read[2].fvalue);
  row_iter = nullptr; row_iter_read = nullptr;
}
