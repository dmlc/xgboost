// Copyright by Contributors
#include <gtest/gtest.h>
#include <dmlc/filesystem.h>

#include <xgboost/data.h>
#include <xgboost/json.h>
#include <xgboost/c_api.h>

#include "../../../src/data/simple_csr_source.h"

#include "../helpers.h"

namespace xgboost {

TEST(SimpleCSRSource, SaveLoadBinary) {
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/simple.libsvm";
  CreateSimpleTestData(tmp_file);
  xgboost::DMatrix * dmat = xgboost::DMatrix::Load(tmp_file, true, false);

  const std::string tmp_binfile = tempdir.path + "/csr_source.binary";
  dmat->SaveToLocalFile(tmp_binfile);
  xgboost::DMatrix * dmat_read = xgboost::DMatrix::Load(tmp_binfile, true, false);

  EXPECT_EQ(dmat->Info().num_col_, dmat_read->Info().num_col_);
  EXPECT_EQ(dmat->Info().num_row_, dmat_read->Info().num_row_);
  EXPECT_EQ(dmat->Info().num_row_, dmat_read->Info().num_row_);

  // Test we have non-empty batch
  EXPECT_EQ(dmat->GetBatches<xgboost::SparsePage>().begin().AtEnd(), false);

  auto row_iter = dmat->GetBatches<xgboost::SparsePage>().begin();
  auto row_iter_read = dmat_read->GetBatches<xgboost::SparsePage>().begin();
  // Test the data read into the first row
  auto first_row = (*row_iter)[0];
  auto first_row_read = (*row_iter_read)[0];
  EXPECT_EQ(first_row.size(), first_row_read.size());
  EXPECT_EQ(first_row[2].index, first_row_read[2].index);
  EXPECT_EQ(first_row[2].fvalue, first_row_read[2].fvalue);
  delete dmat;
  delete dmat_read;
}

TEST(SimpleCSRSource, DataIter) {
  size_t constexpr kRows{16}, kCols{12}, kMissing {2};

  std::vector<float> data(kRows * kCols);
  std::vector<uint32_t> index(kRows * kCols);
#if defined(__APPLE__)
  std::vector<long> offset;
#else
  std::vector<int64_t> offset;
#endif  // defined(__APPLE__)

  for (size_t i = 0; i < kRows * kCols; ++i) {
    data[i] = static_cast<float>(i);
    index[i] = i % kRows;
  }
  for (int64_t i = 0; i < kRows; ++i) {
    offset.push_back(i * static_cast<int64_t>(kRows));
  }

  std::vector<float> labels(kRows);
  for (size_t i = 0; i < kRows; ++i) {
    labels[i] = static_cast<float>(i);
  }

  // define first two rows each having a missing value
  data[13] = std::numeric_limits<float>::quiet_NaN();
  data[23] = std::numeric_limits<float>::quiet_NaN();

  size_t constexpr kEntries = kRows * kCols - kMissing;

  XGBoostBatchCSR block;

  block.size = kRows;
  block.offset = offset.data();
  block.value = data.data();
  block.label = labels.data();

  block.weight = nullptr;

  DataIterHandle iter;
  // XGBoostNativeDataIterSetData;
  // XGDMatrixCreateFromDataIterEx(iter, NAN, , nullptr, nullptr);
  // XGDMatrixCreateFromDataIterEx()
}

}  // namespace xgboost
