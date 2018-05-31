
/*!
 * Copyright 2017 XGBoost contributors
 */
#include <thrust/device_vector.h>
#include <xgboost/base.h>
#include "../helpers.h"
#include "gtest/gtest.h"

#include "../../../src/data/sparse_page_source.h"
#include "../../../src/gbm/gbtree_model.h"
#include "../../../src/tree/updater_gpu_hist.cu"

namespace xgboost {
namespace tree {
TEST(gpu_hist_experimental, TestSparseShard) {
  int rows = 100;
  int columns = 80;
  int max_bins = 4;
  auto dmat = CreateDMatrix(rows, columns, 0.9f);
  common::HistCutMatrix hmat;
  common::GHistIndexMatrix gmat;
  hmat.Init(dmat.get(), max_bins);
  gmat.cut = &hmat;
  gmat.Init(dmat.get());
  TrainParam p;
  p.max_depth = 6;

  dmlc::DataIter<RowBatch>* iter = dmat->RowIterator();
  iter->BeforeFirst();
  CHECK(iter->Next());
  const RowBatch& batch = iter->Value();
  DeviceShard shard(0, 0, 0, rows, hmat.row_ptr.back(), p);
  shard.Init(hmat, batch);
  CHECK(!iter->Next());

  ASSERT_LT(shard.row_stride, columns);

  auto host_gidx_buffer = shard.gidx_buffer.AsVector();

  common::CompressedIterator<uint32_t> gidx(host_gidx_buffer.data(),
                                            hmat.row_ptr.back() + 1);

  for (int i = 0; i < rows; i++) {
    int row_offset = 0;
    for (auto j = gmat.row_ptr[i]; j < gmat.row_ptr[i + 1]; j++) {
      ASSERT_EQ(gidx[i * shard.row_stride + row_offset], gmat.index[j]);
      row_offset++;
    }

    for (; row_offset < shard.row_stride; row_offset++) {
      ASSERT_EQ(gidx[i * shard.row_stride + row_offset], shard.null_gidx_value);
    }
  }
}

TEST(gpu_hist_experimental, TestDenseShard) {
  int rows = 100;
  int columns = 80;
  int max_bins = 4;
  auto dmat = CreateDMatrix(rows, columns, 0);
  common::HistCutMatrix hmat;
  common::GHistIndexMatrix gmat;
  hmat.Init(dmat.get(), max_bins);
  gmat.cut = &hmat;
  gmat.Init(dmat.get());
  TrainParam p;
  p.max_depth = 6;

  dmlc::DataIter<RowBatch>* iter = dmat->RowIterator();
  iter->BeforeFirst();
  CHECK(iter->Next());
  const RowBatch& batch = iter->Value();

  DeviceShard shard(0, 0, 0, rows, hmat.row_ptr.back(), p);
  shard.Init(hmat, batch);
  CHECK(!iter->Next());

  ASSERT_EQ(shard.row_stride, columns);

  auto host_gidx_buffer = shard.gidx_buffer.AsVector();

  common::CompressedIterator<uint32_t> gidx(host_gidx_buffer.data(),
                                            hmat.row_ptr.back() + 1);

  for (int i = 0; i < gmat.index.size(); i++) {
    ASSERT_EQ(gidx[i], gmat.index[i]);
  }
}

}  // namespace tree
}  // namespace xgboost
