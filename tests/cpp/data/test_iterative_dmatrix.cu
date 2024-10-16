/**
 * Copyright 2020-2024, XGBoost contributors
 */
#include <gtest/gtest.h>

#include "../../../src/data/device_adapter.cuh"
#include "../../../src/data/ellpack_page.cuh"
#include "../../../src/data/ellpack_page.h"
#include "../../../src/data/iterative_dmatrix.h"
#include "../../../src/tree/param.h"  // TrainParam
#include "../helpers.h"
#include "test_iterative_dmatrix.h"

namespace xgboost::data {
void TestEquivalent(float sparsity) {
  auto ctx = MakeCUDACtx(0);

  CudaArrayIterForTest iter{sparsity};
  IterativeDMatrix m{&iter, iter.Proxy(), nullptr, Reset, Next,
                     std::numeric_limits<float>::quiet_NaN(), 0, 256,
                     std::numeric_limits<std::int64_t>::max()};
  std::size_t offset = 0;
  auto first = (*m.GetEllpackBatches(&ctx, {}).begin()).Impl();
  std::unique_ptr<EllpackPageImpl> page_concatenated{new EllpackPageImpl{
      &ctx, first->CutsShared(), first->is_dense, first->info.row_stride, 1000 * 100}};
  for (auto& batch : m.GetBatches<EllpackPage>(&ctx, {})) {
    auto page = batch.Impl();
    size_t num_elements = page_concatenated->Copy(&ctx, page, offset);
    offset += num_elements;
  }
  std::vector<common::CompressedByteT> h_iter_buffer;
  auto from_iter = page_concatenated->GetHostAccessor(&ctx, &h_iter_buffer);
  ASSERT_EQ(m.Info().num_col_, CudaArrayIterForTest::Cols());
  ASSERT_EQ(m.Info().num_row_, CudaArrayIterForTest::Rows());

  std::string interface_str = iter.AsArray();
  auto adapter = CupyAdapter(interface_str);
  std::unique_ptr<DMatrix> dm{
      DMatrix::Create(&adapter, std::numeric_limits<float>::quiet_NaN(), 0)};
  auto bp = BatchParam{256, tree::TrainParam::DftSparseThreshold()};
  for (auto& ellpack : dm->GetBatches<EllpackPage>(&ctx, bp)) {
    std::vector<common::CompressedByteT> h_data_buffer;
    auto from_data = ellpack.Impl()->GetHostAccessor(&ctx, &h_data_buffer);

    ASSERT_EQ(from_iter.gidx_fvalue_map.size(), from_data.gidx_fvalue_map.size());
    for (size_t i = 0; i < from_iter.gidx_fvalue_map.size(); ++i) {
      EXPECT_NEAR(from_iter.gidx_fvalue_map[i], from_data.gidx_fvalue_map[i], kRtEps);
    }
    ASSERT_EQ(from_iter.min_fvalue.size(), from_data.min_fvalue.size());
    for (size_t i = 0; i < from_iter.min_fvalue.size(); ++i) {
      ASSERT_NEAR(from_iter.min_fvalue[i], from_data.min_fvalue[i], kRtEps);
    }
    ASSERT_EQ(from_iter.NumFeatures(), from_data.NumFeatures());
    for (size_t i = 0; i < from_iter.NumFeatures() + 1; ++i) {
      ASSERT_EQ(from_iter.feature_segments[i], from_data.feature_segments[i]);
    }

    std::vector<common::CompressedByteT> buffer_from_iter, buffer_from_data;
    auto data_iter = page_concatenated->GetHostAccessor(&ctx, &buffer_from_iter);
    auto data_buf = ellpack.Impl()->GetHostAccessor(&ctx, &buffer_from_data);
    ASSERT_NE(buffer_from_data.size(), 0);
    ASSERT_NE(buffer_from_iter.size(), 0);
    CHECK_EQ(ellpack.Impl()->NumSymbols(), page_concatenated->NumSymbols());
    CHECK_EQ(from_data.n_rows * from_data.row_stride, from_data.n_rows * from_iter.row_stride);
    for (size_t i = 0; i < from_data.n_rows * from_data.row_stride; ++i) {
      CHECK_EQ(data_buf.gidx_iter[i], data_iter.gidx_iter[i]);
    }
  }
}

TEST(IterativeDeviceDMatrix, Basic) {
  TestEquivalent(0.0);
  TestEquivalent(0.5);
}

TEST(IterativeDeviceDMatrix, RowMajor) {
  CudaArrayIterForTest iter(0.0f);
  IterativeDMatrix m{&iter, iter.Proxy(), nullptr,
                     Reset, Next,         std::numeric_limits<float>::quiet_NaN(),
                     0,     256,          std::numeric_limits<std::int64_t>::max()};
  size_t n_batches = 0;
  std::string interface_str = iter.AsArray();
  Context ctx{MakeCUDACtx(0)};
  for (auto& ellpack : m.GetBatches<EllpackPage>(&ctx, {})) {
    n_batches ++;
    auto impl = ellpack.Impl();
    std::vector<common::CompressedByteT> h_gidx;
    auto h_accessor = impl->GetHostAccessor(&ctx, &h_gidx);
    auto cols = CudaArrayIterForTest::Cols();
    auto rows = CudaArrayIterForTest::Rows();

    auto j_interface =
        Json::Load({interface_str.c_str(), interface_str.size()});
    ArrayInterface<2> loaded {get<Object const>(j_interface)};
    std::vector<float> h_data(cols * rows);
    common::Span<float const> s_data{static_cast<float const*>(loaded.data), cols * rows};
    dh::CopyDeviceSpanToVector(&h_data, s_data);

    auto cut_ptr = h_accessor.feature_segments;
    for (auto i = 0ull; i < rows * cols; i++) {
      int column_idx = i % cols;
      EXPECT_EQ(impl->Cuts().SearchBin(h_data[i], column_idx),
                h_accessor.gidx_iter[i] + cut_ptr[column_idx]);
    }
    EXPECT_EQ(m.Info().num_col_, cols);
    EXPECT_EQ(m.Info().num_row_, rows);
    EXPECT_EQ(m.Info().num_nonzero_, rows * cols);
  }
  // All batches are concatenated.
  ASSERT_EQ(n_batches, 1);
}

TEST(IterativeDeviceDMatrix, RowMajorMissing) {
  const float kMissing = std::numeric_limits<float>::quiet_NaN();
  bst_idx_t rows = 4;
  size_t cols = 3;
  CudaArrayIterForTest iter{0.0f, rows, cols, 2};
  std::string interface_str = iter.AsArray();
  auto j_interface = Json::Load({interface_str.c_str(), interface_str.size()});
  ArrayInterface<2> loaded{get<Object const>(j_interface)};
  std::vector<float> h_data(cols * rows);
  common::Span<float const> s_data{static_cast<float const*>(loaded.data), cols * rows};
  dh::CopyDeviceSpanToVector(&h_data, s_data);
  h_data[1] = kMissing;
  h_data[5] = kMissing;
  h_data[6] = kMissing;
  h_data[9] = kMissing;  // idx = (2, 0)
  h_data[10] = kMissing; // idx = (2, 1)
  auto ptr =
      thrust::device_ptr<float>(reinterpret_cast<float*>(get<Integer>(j_interface["data"][0])));
  thrust::copy(h_data.cbegin(), h_data.cend(), ptr);
  IterativeDMatrix m{&iter, iter.Proxy(), nullptr,
                     Reset, Next,         std::numeric_limits<float>::quiet_NaN(),
                     0,     256,          std::numeric_limits<std::int64_t>::max()};
  auto ctx = MakeCUDACtx(0);
  auto& ellpack =
      *m.GetBatches<EllpackPage>(&ctx, BatchParam{256, tree::TrainParam::DftSparseThreshold()})
           .begin();
  auto impl = ellpack.Impl();
  std::vector<common::CompressedByteT> h_gidx;
  auto h_acc = impl->GetHostAccessor(&ctx, &h_gidx);
  // null values get placed after valid values in a row
  ASSERT_FALSE(h_acc.IsDenseCompressed());
  ASSERT_EQ(h_acc.row_stride, cols - 1);
  ASSERT_EQ(h_acc.gidx_iter[7], impl->GetDeviceAccessor(&ctx).NullValue());
  for (std::size_t i = 0; i < 7; ++i) {
  ASSERT_NE(h_acc.gidx_iter[i], impl->GetDeviceAccessor(&ctx).NullValue());
  }

  EXPECT_EQ(m.Info().num_col_, cols);
  EXPECT_EQ(m.Info().num_row_, rows);
  EXPECT_EQ(m.Info().num_nonzero_, rows * cols - 5);
}

TEST(IterativeDeviceDMatrix, IsDense) {
  int num_bins = 16;
  auto test = [num_bins](float sparsity) {
    CudaArrayIterForTest iter(sparsity);
    IterativeDMatrix m(&iter, iter.Proxy(), nullptr, Reset, Next,
                       std::numeric_limits<float>::quiet_NaN(), 0, num_bins,
                       std::numeric_limits<std::int64_t>::max());
    if (sparsity == 0.0) {
      ASSERT_TRUE(m.IsDense());
    } else {
      ASSERT_FALSE(m.IsDense());
    }
  };
  test(0.0);
  test(0.1);
  test(1.0);
}

TEST(IterativeDeviceDMatrix, Ref) {
  Context ctx{MakeCUDACtx(0)};
  TestRefDMatrix<EllpackPage, CudaArrayIterForTest>(
      &ctx, [](EllpackPage const& page) { return page.Impl()->Cuts(); });
}
}  // namespace xgboost::data
