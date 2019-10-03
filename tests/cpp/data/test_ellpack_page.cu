/*!
 * Copyright 2019 XGBoost contributors
 */
#include <xgboost/base.h>

#include <utility>

#include "../helpers.h"
#include "gtest/gtest.h"

#include "../../../src/common/hist_util.h"
#include "../../../src/data/ellpack_page.cuh"

namespace xgboost {

TEST(EllpackPage, EmptyDMatrix) {
  constexpr int kNRows = 0, kNCols = 0, kMaxBin = 256, kGpuBatchNRows = 64;
  constexpr float kSparsity = 0;
  auto dmat = *CreateDMatrix(kNRows, kNCols, kSparsity);
  auto& page = *dmat->GetBatches<EllpackPage>({0, kMaxBin, kGpuBatchNRows}).begin();
  auto impl = page.Impl();
  ASSERT_EQ(impl->matrix.info.feature_segments.size(), 1);
  ASSERT_EQ(impl->matrix.info.min_fvalue.size(), 0);
  ASSERT_EQ(impl->matrix.info.gidx_fvalue_map.size(), 0);
  ASSERT_EQ(impl->matrix.info.row_stride, 0);
  ASSERT_EQ(impl->matrix.info.n_bins, 0);
  ASSERT_EQ(impl->gidx_buffer.size(), 4);
}

TEST(EllpackPage, BuildGidxDense) {
  int constexpr kNRows = 16, kNCols = 8;
  auto page = BuildEllpackPage(kNRows, kNCols);

  std::vector<common::CompressedByteT> h_gidx_buffer(page->gidx_buffer.size());
  dh::CopyDeviceSpanToVector(&h_gidx_buffer, page->gidx_buffer);
  common::CompressedIterator<uint32_t> gidx(h_gidx_buffer.data(), 25);

  ASSERT_EQ(page->matrix.info.row_stride, kNCols);

  std::vector<uint32_t> solution = {
    0, 3, 8,  9, 14, 17, 20, 21,
    0, 4, 7, 10, 14, 16, 19, 22,
    1, 3, 7, 11, 14, 15, 19, 21,
    2, 3, 7,  9, 13, 16, 20, 22,
    2, 3, 6,  9, 12, 16, 20, 21,
    1, 5, 6, 10, 13, 16, 20, 21,
    2, 5, 8,  9, 13, 17, 19, 22,
    2, 4, 6, 10, 14, 17, 19, 21,
    2, 5, 7,  9, 13, 16, 19, 22,
    0, 3, 8, 10, 12, 16, 19, 22,
    1, 3, 7, 10, 13, 16, 19, 21,
    1, 3, 8, 10, 13, 17, 20, 22,
    2, 4, 6,  9, 14, 15, 19, 22,
    1, 4, 6,  9, 13, 16, 19, 21,
    2, 4, 8, 10, 14, 15, 19, 22,
    1, 4, 7, 10, 14, 16, 19, 21,
  };
  for (size_t i = 0; i < kNRows * kNCols; ++i) {
    ASSERT_EQ(solution[i], gidx[i]);
  }
}

TEST(EllpackPage, BuildGidxSparse) {
  int constexpr kNRows = 16, kNCols = 8;
  auto page = BuildEllpackPage(kNRows, kNCols, 0.9f);

  std::vector<common::CompressedByteT> h_gidx_buffer(page->gidx_buffer.size());
  dh::CopyDeviceSpanToVector(&h_gidx_buffer, page->gidx_buffer);
  common::CompressedIterator<uint32_t> gidx(h_gidx_buffer.data(), 25);

  ASSERT_LE(page->matrix.info.row_stride, 3);

  // row_stride = 3, 16 rows, 48 entries for ELLPack
  std::vector<uint32_t> solution = {
    15, 24, 24,  0, 24, 24, 24, 24, 24, 24, 24, 24, 20, 24, 24, 24,
    24, 24, 24, 24, 24,  5, 24, 24,  0, 16, 24, 15, 24, 24, 24, 24,
    24,  7, 14, 16,  4, 24, 24, 24, 24, 24,  9, 24, 24,  1, 24, 24
  };
  for (size_t i = 0; i < kNRows * page->matrix.info.row_stride; ++i) {
    ASSERT_EQ(solution[i], gidx[i]);
  }
}

}  // namespace xgboost
