// Copyright (c) 2019 by Contributors
#include <gtest/gtest.h>
#include <xgboost/c_api.h>
#include <xgboost/data.h>
#include <xgboost/version_config.h>
#include "../../../src/c_api/adapter.h"
#include "../../../src/data/simple_dmatrix.h"
#include "../../../src/common/timer.h"
using namespace xgboost;  // NOLINT
TEST(c_api, CSRAdapter) {
  int m = 3;
  int n = 2;
  std::vector<float> data = {1, 2, 3, 4, 5};
  std::vector<unsigned> feature_idx = {0, 1, 0, 1, 1};
  std::vector<size_t> row_ptr = {0, 2, 4, 5};
  CSRAdapter adapter(row_ptr.data(), feature_idx.data(), data.data(),
                     row_ptr.size() - 1, data.size(), n);
  auto batch0 = adapter[0];
  EXPECT_EQ(batch0.GetElement(0).value, 1);
  EXPECT_EQ(batch0.GetElement(1).value, 2);

  auto batch1 = adapter[1];
  EXPECT_EQ(batch1.GetElement(0).value, 3);
  EXPECT_EQ(batch1.GetElement(1).value, 4);
  auto batch2 = adapter[2];
  EXPECT_EQ(batch2.GetElement(0).value, 5);
  EXPECT_EQ(batch2.GetElement(0).row_idx, 2);
  EXPECT_EQ(batch2.GetElement(0).column_idx, 1);

  data::SimpleDMatrix dmat(adapter, -1, std::nan(""));
  EXPECT_EQ(dmat.Info().num_col_, 2);
  EXPECT_EQ(dmat.Info().num_row_, 3);
  EXPECT_EQ(dmat.Info().num_nonzero_, 5);

  for (auto &batch : dmat.GetBatches<SparsePage>()) {
    for (auto i = 0ull; i < batch.Size(); i++) {
      auto inst = batch[i];
      for(auto j = 0ull; j < inst.size(); j++)
      {
        EXPECT_EQ(inst[j].fvalue, data[row_ptr[i] + j]);
        EXPECT_EQ(inst[j].index, feature_idx[row_ptr[i] + j]);
      }
    }
  }
}
TEST(c_api, DenseAdapter) {
  int m = 3;
  int n = 2;
  std::vector<float> data = {1, 2, 3, 4, 5, 6};
  DenseAdapter adapter(data.data(), m, m*n, n);
  data::SimpleDMatrix dmat(adapter,-1,std::numeric_limits<float>::quiet_NaN());
  EXPECT_EQ(dmat.Info().num_col_, 2);
  EXPECT_EQ(dmat.Info().num_row_, 3);
  EXPECT_EQ(dmat.Info().num_nonzero_, 6);

  for (auto &batch : dmat.GetBatches<SparsePage>()) {
    for (auto i = 0ull; i < batch.Size(); i++) {
      auto inst = batch[i];
      for(auto j = 0ull; j < inst.size(); j++)
      {
        EXPECT_EQ(inst[j].fvalue, data[i*n+j]);
        EXPECT_EQ(inst[j].index, j);
      }
    }
  }
}

TEST(c_api, CSCAdapter) {
  std::vector<float> data = {1, 3, 2, 4, 5};
  std::vector<unsigned> row_idx = {0, 1, 0, 1, 2};
  std::vector<size_t> col_ptr = {0, 2, 5};
  CSCAdapter adapter(col_ptr.data(),row_idx.data(),data.data(),3,5,2);
  data::SimpleDMatrix dmat(adapter,-1,std::numeric_limits<float>::quiet_NaN());
  EXPECT_EQ(dmat.Info().num_col_, 2);
  EXPECT_EQ(dmat.Info().num_row_, 3);
  EXPECT_EQ(dmat.Info().num_nonzero_, 5);

  auto &batch = *dmat.GetBatches<SparsePage>().begin();
  auto inst = batch[0];
  EXPECT_EQ(inst[0].fvalue, 1);
  EXPECT_EQ(inst[0].index, 0);
  EXPECT_EQ(inst[1].fvalue, 2);
  EXPECT_EQ(inst[1].index, 1);

  inst = batch[1];
  EXPECT_EQ(inst[0].fvalue, 3);
  EXPECT_EQ(inst[0].index, 0);
  EXPECT_EQ(inst[1].fvalue, 4);
  EXPECT_EQ(inst[1].index, 1);

  inst = batch[2];
  EXPECT_EQ(inst[0].fvalue, 5);
  EXPECT_EQ(inst[0].index, 1);
}

