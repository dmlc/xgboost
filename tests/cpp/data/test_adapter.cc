// Copyright (c) 2019 by Contributors
#include <gtest/gtest.h>
#include <xgboost/data.h>
#include "../../../src/data/adapter.h"
#include "../../../src/data/simple_dmatrix.h"
#include "../../../src/common/timer.h"
#include "../helpers.h"
using namespace xgboost;  // NOLINT
TEST(adapter, CSRAdapter) {
  int n = 2;
  std::vector<float> data = {1, 2, 3, 4, 5};
  std::vector<unsigned> feature_idx = {0, 1, 0, 1, 1};
  std::vector<size_t> row_ptr = {0, 2, 4, 5};
  data::CSRAdapter adapter(row_ptr.data(), feature_idx.data(), data.data(),
                     row_ptr.size() - 1, data.size(), n);
  adapter.Next();
  auto & batch = adapter.Value();
  auto line0 = batch.GetLine(0);
  EXPECT_EQ(line0.GetElement(0).value, 1);
  EXPECT_EQ(line0.GetElement(1).value, 2);

  auto line1 = batch.GetLine(1);
  EXPECT_EQ(line1 .GetElement(0).value, 3);
  EXPECT_EQ(line1 .GetElement(1).value, 4);
  auto line2 = batch.GetLine(2);
  EXPECT_EQ(line2 .GetElement(0).value, 5);
  EXPECT_EQ(line2 .GetElement(0).row_idx, 2);
  EXPECT_EQ(line2 .GetElement(0).column_idx, 1);
}

TEST(adapter, CSCAdapterColsMoreThanRows) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<unsigned> row_idx = {0, 1, 0, 1, 0, 1, 0, 1};
  std::vector<size_t> col_ptr = {0, 2, 4, 6, 8};
  // Infer row count
  data::CSCAdapter adapter(col_ptr.data(), row_idx.data(), data.data(), 4, 0);
  data::SimpleDMatrix dmat(&adapter, std::numeric_limits<float>::quiet_NaN(), -1);
  EXPECT_EQ(dmat.Info().num_col_, 4);
  EXPECT_EQ(dmat.Info().num_row_, 2);
  EXPECT_EQ(dmat.Info().num_nonzero_, 8);

  auto &batch = *dmat.GetBatches<SparsePage>().begin();
  auto inst = batch[0];
  EXPECT_EQ(inst[0].fvalue, 1);
  EXPECT_EQ(inst[0].index, 0);
  EXPECT_EQ(inst[1].fvalue, 3);
  EXPECT_EQ(inst[1].index, 1);
  EXPECT_EQ(inst[2].fvalue, 5);
  EXPECT_EQ(inst[2].index, 2);
  EXPECT_EQ(inst[3].fvalue, 7);
  EXPECT_EQ(inst[3].index, 3);

  inst = batch[1];
  EXPECT_EQ(inst[0].fvalue, 2);
  EXPECT_EQ(inst[0].index, 0);
  EXPECT_EQ(inst[1].fvalue, 4);
  EXPECT_EQ(inst[1].index, 1);
  EXPECT_EQ(inst[2].fvalue, 6);
  EXPECT_EQ(inst[2].index, 2);
  EXPECT_EQ(inst[3].fvalue, 8);
  EXPECT_EQ(inst[3].index, 3);
}

TEST(c_api, DMatrixSliceAdapterFromSimpleDMatrix) {
  auto pp_dmat = CreateDMatrix(6, 2, 1.0);
  auto p_dmat = *pp_dmat;

  std::vector<int> ridx_set = {1, 3, 5};
  data::DMatrixSliceAdapter adapter(p_dmat.get(),
                                    {ridx_set.data(), ridx_set.size()});
  EXPECT_EQ(adapter.NumRows(), ridx_set.size());

  adapter.BeforeFirst();
  for (auto &batch : p_dmat->GetBatches<SparsePage>()) {
    adapter.Next();
    auto &adapter_batch = adapter.Value();
    for (auto i = 0ull; i < adapter_batch.Size(); i++) {
      auto inst = batch[ridx_set[i]];
      auto line = adapter_batch.GetLine(i);
      ASSERT_EQ(inst.size(), line.Size());
      for (auto j = 0ull; j < line.Size(); j++) {
        EXPECT_EQ(inst[j].fvalue, line.GetElement(j).value);
        EXPECT_EQ(inst[j].index, line.GetElement(j).column_idx);
        EXPECT_EQ(i, line.GetElement(j).row_idx);
      }
    }
  }

  delete pp_dmat;
}
