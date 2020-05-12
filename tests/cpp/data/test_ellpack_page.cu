/*!
 * Copyright 2019 XGBoost contributors
 */
#include <xgboost/base.h>

#include <utility>

#include "../helpers.h"
#include "../histogram_helpers.h"
#include "gtest/gtest.h"

#include "../../../src/common/hist_util.h"
#include "../../../src/data/ellpack_page.cuh"

namespace xgboost {

TEST(EllpackPage, EmptyDMatrix) {
  constexpr int kNRows = 0, kNCols = 0, kMaxBin = 256;
  constexpr float kSparsity = 0;
  auto dmat = RandomDataGenerator(kNRows, kNCols, kSparsity).GenerateDMatrix();
  auto& page = *dmat->GetBatches<EllpackPage>({0, kMaxBin}).begin();
  auto impl = page.Impl();
  ASSERT_EQ(impl->row_stride, 0);
  ASSERT_EQ(impl->Cuts().TotalBins(), 0);
  ASSERT_EQ(impl->gidx_buffer.Size(), 4);
}

TEST(EllpackPage, BuildGidxDense) {
  int constexpr kNRows = 16, kNCols = 8;
  auto page = BuildEllpackPage(kNRows, kNCols);

  std::vector<common::CompressedByteT> h_gidx_buffer(page->gidx_buffer.HostVector());
  common::CompressedIterator<uint32_t> gidx(h_gidx_buffer.data(), page->NumSymbols());

  ASSERT_EQ(page->row_stride, kNCols);

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

  std::vector<common::CompressedByteT> h_gidx_buffer(page->gidx_buffer.HostVector());
  common::CompressedIterator<uint32_t> gidx(h_gidx_buffer.data(), 25);

  ASSERT_LE(page->row_stride, 3);

  // row_stride = 3, 16 rows, 48 entries for ELLPack
  std::vector<uint32_t> solution = {
    15, 24, 24,  0, 24, 24, 24, 24, 24, 24, 24, 24, 20, 24, 24, 24,
    24, 24, 24, 24, 24,  5, 24, 24,  0, 16, 24, 15, 24, 24, 24, 24,
    24,  7, 14, 16,  4, 24, 24, 24, 24, 24,  9, 24, 24,  1, 24, 24
  };
  for (size_t i = 0; i < kNRows * page->row_stride; ++i) {
    ASSERT_EQ(solution[i], gidx[i]);
  }
}

struct ReadRowFunction {
  EllpackDeviceAccessor matrix;
  int row;
  bst_float* row_data_d;
  ReadRowFunction(EllpackDeviceAccessor matrix, int row, bst_float* row_data_d)
      : matrix(std::move(matrix)), row(row), row_data_d(row_data_d) {}

  __device__ void operator()(size_t col) {
    auto value = matrix.GetFvalue(row, col);
    if (isnan(value)) {
      value = -1;
    }
    row_data_d[col] = value;
  }
};

TEST(EllpackPage, Copy) {
  constexpr size_t kRows = 1024;
  constexpr size_t kCols = 16;
  constexpr size_t kPageSize = 1024;

  // Create a DMatrix with multiple batches.
  dmlc::TemporaryDirectory tmpdir;
  std::unique_ptr<DMatrix>
      dmat(CreateSparsePageDMatrixWithRC(kRows, kCols, kPageSize, true, tmpdir));
  BatchParam param{0, 256, kPageSize};
  auto page = (*dmat->GetBatches<EllpackPage>(param).begin()).Impl();

  // Create an empty result page.
  EllpackPageImpl result(0, page->Cuts(), page->is_dense, page->row_stride,
                         kRows);

  // Copy batch pages into the result page.
  size_t offset = 0;
  for (auto& batch : dmat->GetBatches<EllpackPage>(param)) {
    size_t num_elements = result.Copy(0, batch.Impl(), offset);
    offset += num_elements;
  }

  size_t current_row = 0;
  thrust::device_vector<bst_float> row_d(kCols);
  thrust::device_vector<bst_float> row_result_d(kCols);
  std::vector<bst_float> row(kCols);
  std::vector<bst_float> row_result(kCols);
  for (auto& page : dmat->GetBatches<EllpackPage>(param)) {
    auto impl = page.Impl();
    EXPECT_EQ(impl->base_rowid, current_row);

    for (size_t i = 0; i < impl->Size(); i++) {
      dh::LaunchN(0, kCols, ReadRowFunction(impl->GetDeviceAccessor(0), current_row, row_d.data().get()));
      thrust::copy(row_d.begin(), row_d.end(), row.begin());

      dh::LaunchN(0, kCols, ReadRowFunction(result.GetDeviceAccessor(0), current_row, row_result_d.data().get()));
      thrust::copy(row_result_d.begin(), row_result_d.end(), row_result.begin());

      EXPECT_EQ(row, row_result);
      current_row++;
    }
  }
}

TEST(EllpackPage, Compact) {
  constexpr size_t kRows = 16;
  constexpr size_t kCols = 2;
  constexpr size_t kPageSize = 1;
  constexpr size_t kCompactedRows = 8;

  // Create a DMatrix with multiple batches.
  dmlc::TemporaryDirectory tmpdir;
  std::unique_ptr<DMatrix>
      dmat(CreateSparsePageDMatrixWithRC(kRows, kCols, kPageSize, true, tmpdir));
  BatchParam param{0, 256, kPageSize};
  auto page = (*dmat->GetBatches<EllpackPage>(param).begin()).Impl();

  // Create an empty result page.
  EllpackPageImpl result(0, page->Cuts(), page->is_dense, page->row_stride,
                         kCompactedRows);

  // Compact batch pages into the result page.
  std::vector<size_t> row_indexes_h {
    SIZE_MAX, 0, 1, 2, SIZE_MAX, 3, SIZE_MAX, 4, 5, SIZE_MAX, 6, SIZE_MAX, 7, SIZE_MAX, SIZE_MAX,
    SIZE_MAX};
  thrust::device_vector<size_t> row_indexes_d = row_indexes_h;
  common::Span<size_t> row_indexes_span(row_indexes_d.data().get(), kRows);
  for (auto& batch : dmat->GetBatches<EllpackPage>(param)) {
    result.Compact(0, batch.Impl(), row_indexes_span);
  }

  size_t current_row = 0;
  thrust::device_vector<bst_float> row_d(kCols);
  thrust::device_vector<bst_float> row_result_d(kCols);
  std::vector<bst_float> row(kCols);
  std::vector<bst_float> row_result(kCols);
  for (auto& page : dmat->GetBatches<EllpackPage>(param)) {
    auto impl = page.Impl();
    EXPECT_EQ(impl->base_rowid, current_row);

    for (size_t i = 0; i < impl->Size(); i++) {
      size_t compacted_row = row_indexes_h[current_row];
      if (compacted_row == SIZE_MAX) {
        current_row++;
        continue;
      }

      dh::LaunchN(0, kCols, ReadRowFunction(impl->GetDeviceAccessor(0), current_row, row_d.data().get()));
      dh::safe_cuda (cudaDeviceSynchronize());
      thrust::copy(row_d.begin(), row_d.end(), row.begin());

      dh::LaunchN(0, kCols,
                  ReadRowFunction(result.GetDeviceAccessor(0), compacted_row, row_result_d.data().get()));
      thrust::copy(row_result_d.begin(), row_result_d.end(), row_result.begin());

      EXPECT_EQ(row, row_result);
      current_row++;
    }
  }
}

}  // namespace xgboost
