/*!
 * Copyright 2020 XGBoost contributors
 */
#include <gtest/gtest.h>

#include "../helpers.h"
#include "../../../src/data/iterative_device_dmatrix.h"
#include "../../../src/data/ellpack_page.cuh"
#include "../../../src/data/device_adapter.cuh"

namespace xgboost {
namespace data {

void TestEquivalent(float sparsity) {
  CudaArrayIterForTest iter{sparsity};
  IterativeDeviceDMatrix m(
      &iter, iter.Proxy(), Reset, Next, std::numeric_limits<float>::quiet_NaN(),
      0, 256);
  size_t offset = 0;
  auto first = (*m.GetEllpackBatches({}).begin()).Impl();
  std::unique_ptr<EllpackPageImpl> page_concatenated {
    new EllpackPageImpl(0, first->Cuts(), first->is_dense,
                        first->row_stride, 1000 * 100)};
  for (auto& batch : m.GetBatches<EllpackPage>()) {
    auto page = batch.Impl();
    size_t num_elements = page_concatenated->Copy(0, page, offset);
    offset += num_elements;
  }
  auto from_iter = page_concatenated->GetDeviceAccessor(0);
  ASSERT_EQ(m.Info().num_col_, CudaArrayIterForTest::kCols);
  ASSERT_EQ(m.Info().num_row_, CudaArrayIterForTest::kRows);

  std::string interface_str = iter.AsArray();
  auto adapter = CupyAdapter(interface_str);
  std::unique_ptr<DMatrix> dm{
      DMatrix::Create(&adapter, std::numeric_limits<float>::quiet_NaN(), 0)};
  BatchParam bp {0, 256};
  for (auto& ellpack : dm->GetBatches<EllpackPage>(bp)) {
    auto from_data = ellpack.Impl()->GetDeviceAccessor(0);

    std::vector<float> cuts_from_iter(from_iter.gidx_fvalue_map.size());
    std::vector<uint32_t> cut_ptrs_iter(from_iter.feature_segments.size());
    dh::CopyDeviceSpanToVector(&cuts_from_iter, from_iter.gidx_fvalue_map);
    dh::CopyDeviceSpanToVector(&cut_ptrs_iter, from_iter.feature_segments);

    std::vector<float> cuts_from_data(from_data.gidx_fvalue_map.size());
    std::vector<uint32_t> cut_ptrs_data(from_data.feature_segments.size());
    dh::CopyDeviceSpanToVector(&cuts_from_data, from_data.gidx_fvalue_map);
    dh::CopyDeviceSpanToVector(&cut_ptrs_data, from_data.feature_segments);

    ASSERT_EQ(cuts_from_iter.size(), cuts_from_data.size());
    for (size_t i = 0; i < cuts_from_iter.size(); ++i) {
      EXPECT_NEAR(cuts_from_iter[i], cuts_from_data[i], kRtEps);
    }
    ASSERT_EQ(cut_ptrs_iter.size(), cut_ptrs_data.size());
    for (size_t i = 0; i < cut_ptrs_iter.size(); ++i) {
      ASSERT_EQ(cut_ptrs_iter[i], cut_ptrs_data[i]);
    }

    auto const& buffer_from_iter = page_concatenated->gidx_buffer;
    auto const& buffer_from_data = ellpack.Impl()->gidx_buffer;
    ASSERT_NE(buffer_from_data.Size(), 0);
    ASSERT_EQ(buffer_from_data.ConstHostVector(), buffer_from_data.ConstHostVector());
  }
}

TEST(IterativeDeviceDMatrix, Basic) {
  TestEquivalent(0.0);
  TestEquivalent(0.5);
}

TEST(IterativeDeviceDMatrix, RowMajor) {
  CudaArrayIterForTest iter(0.0f);
  IterativeDeviceDMatrix m(
      &iter, iter.Proxy(), Reset, Next, std::numeric_limits<float>::quiet_NaN(),
      0, 256);
  size_t n_batches = 0;
  std::string interface_str = iter.AsArray();
  for (auto& ellpack : m.GetBatches<EllpackPage>()) {
    n_batches ++;
    auto impl = ellpack.Impl();
    common::CompressedIterator<uint32_t> iterator(
        impl->gidx_buffer.HostVector().data(), impl->NumSymbols());
    auto cols = CudaArrayIterForTest::kCols;
    auto rows = CudaArrayIterForTest::kRows;

    auto j_interface =
        Json::Load({interface_str.c_str(), interface_str.size()});
    ArrayInterface loaded {get<Object const>(j_interface)};
    std::vector<float> h_data(cols * rows);
    common::Span<float> s_data{static_cast<float*>(loaded.data), cols * rows};
    dh::CopyDeviceSpanToVector(&h_data, s_data);

    for(auto i = 0ull; i < rows * cols; i++) {
      int column_idx = i % cols;
      EXPECT_EQ(impl->Cuts().SearchBin(h_data[i], column_idx), iterator[i]);
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
  size_t rows = 10;
  size_t cols = 2;
  CudaArrayIterForTest iter(0.0f, rows, cols, 2);
  std::string interface_str = iter.AsArray();
  auto j_interface =
      Json::Load({interface_str.c_str(), interface_str.size()});
  ArrayInterface loaded {get<Object const>(j_interface)};
  std::vector<float> h_data(cols * rows);
  common::Span<float> s_data{static_cast<float*>(loaded.data), cols * rows};
  dh::CopyDeviceSpanToVector(&h_data, s_data);
  h_data[1] = kMissing;
  h_data[5] = kMissing;
  h_data[6] = kMissing;
  auto ptr = thrust::device_ptr<float>(
      reinterpret_cast<float *>(get<Integer>(j_interface["data"][0])));
  thrust::copy(h_data.cbegin(), h_data.cend(), ptr);

  IterativeDeviceDMatrix m(
      &iter, iter.Proxy(), Reset, Next, std::numeric_limits<float>::quiet_NaN(),
      0, 256);
  auto &ellpack = *m.GetBatches<EllpackPage>({0, 256, 0}).begin();
  auto impl = ellpack.Impl();
  common::CompressedIterator<uint32_t> iterator(
      impl->gidx_buffer.HostVector().data(), impl->NumSymbols());
  EXPECT_EQ(iterator[1], impl->GetDeviceAccessor(0).NullValue());
  EXPECT_EQ(iterator[5], impl->GetDeviceAccessor(0).NullValue());
  // null values get placed after valid values in a row
  EXPECT_EQ(iterator[7], impl->GetDeviceAccessor(0).NullValue());
  EXPECT_EQ(m.Info().num_col_, cols);
  EXPECT_EQ(m.Info().num_row_, rows);
  EXPECT_EQ(m.Info().num_nonzero_, rows* cols - 3);
}

TEST(IterativeDeviceDMatrix, IsDense) {
  int num_bins = 16;
  auto test = [num_bins] (float sparsity) {
    CudaArrayIterForTest iter(sparsity);
    IterativeDeviceDMatrix m(
        &iter, iter.Proxy(), Reset, Next, std::numeric_limits<float>::quiet_NaN(),
        0, 256);
    if (sparsity == 0.0) {
      ASSERT_TRUE(m.IsDense());
    } else {
      ASSERT_FALSE(m.IsDense());
    }
  };
  test(0.0);
  test(0.1);
}
}  // namespace data
}  // namespace xgboost
