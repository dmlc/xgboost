
// Copyright (c) 2019 by Contributors
#include <gtest/gtest.h>
#include <xgboost/data.h>
#include "../../../src/data/adapter.h"
#include "../../../src/data/ellpack_page.cuh"
#include "../../../src/data/device_dmatrix.h"
#include "../helpers.h"
#include <thrust/device_vector.h>
#include "../../../src/data/device_adapter.cuh"
#include "../../../src/gbm/gbtree_model.h"
#include "../common/test_hist_util.h"
#include "../../../src/common/compressed_iterator.h"
#include "../../../src/common/math.h"
#include "test_array_interface.h"
using namespace xgboost;  // NOLINT

TEST(DeviceDMatrix, RowMajor) {
  int num_rows = 1000;
  int num_columns = 50;
  auto x = common::GenerateRandom(num_rows, num_columns);
  auto x_device = thrust::device_vector<float>(x);
  auto adapter = common::AdapterFromData(x_device, num_rows, num_columns);

  data::DeviceDMatrix dmat(&adapter,
                           std::numeric_limits<float>::quiet_NaN(), 1, 256);

  auto &batch = *dmat.GetBatches<EllpackPage>({0, 256, 0}).begin();
  auto impl = batch.Impl();
  common::CompressedIterator<uint32_t> iterator(
      impl->gidx_buffer.HostVector().data(), impl->NumSymbols());
  for(auto i = 0ull; i < x.size(); i++)
  {
    int column_idx = i % num_columns;
    EXPECT_EQ(impl->Cuts().SearchBin(x[i], column_idx), iterator[i]);
  }
  EXPECT_EQ(dmat.Info().num_col_, num_columns);
  EXPECT_EQ(dmat.Info().num_row_, num_rows);
  EXPECT_EQ(dmat.Info().num_nonzero_, num_rows * num_columns);

}

TEST(DeviceDMatrix, RowMajorMissing) {
  const float kMissing = std::numeric_limits<float>::quiet_NaN();
  int num_rows = 10;
  int num_columns = 2;
  auto x = common::GenerateRandom(num_rows, num_columns);
  x[1] = kMissing;
  x[5] = kMissing;
  x[6] = kMissing;
  auto x_device = thrust::device_vector<float>(x);
  auto adapter = common::AdapterFromData(x_device, num_rows, num_columns);

  data::DeviceDMatrix dmat(&adapter, kMissing, 1, 256);

  auto &batch = *dmat.GetBatches<EllpackPage>({0, 256, 0}).begin();
  auto impl = batch.Impl();
  common::CompressedIterator<uint32_t> iterator(
      impl->gidx_buffer.HostVector().data(), impl->NumSymbols());
  EXPECT_EQ(iterator[1], impl->GetDeviceAccessor(0).NullValue());
  EXPECT_EQ(iterator[5], impl->GetDeviceAccessor(0).NullValue());
  // null values get placed after valid values in a row
  EXPECT_EQ(iterator[7], impl->GetDeviceAccessor(0).NullValue());
  EXPECT_EQ(dmat.Info().num_col_, num_columns);
  EXPECT_EQ(dmat.Info().num_row_, num_rows);
  EXPECT_EQ(dmat.Info().num_nonzero_, num_rows*num_columns-3);

}

TEST(DeviceDMatrix, ColumnMajor) {
  constexpr size_t kRows{100};
  std::vector<Json> columns;
  thrust::device_vector<double> d_data_0(kRows);
  thrust::device_vector<uint32_t> d_data_1(kRows);

  columns.emplace_back(GenerateDenseColumn<double>("<f8", kRows, &d_data_0));
  columns.emplace_back(GenerateDenseColumn<uint32_t>("<u4", kRows, &d_data_1));

  Json column_arr{columns};

  std::stringstream ss;
  Json::Dump(column_arr, &ss);
  std::string str = ss.str();

  data::CudfAdapter adapter(str);
  data::DeviceDMatrix dmat(&adapter, std::numeric_limits<float>::quiet_NaN(),
                           -1, 256);
  auto &batch = *dmat.GetBatches<EllpackPage>({0, 256, 0}).begin();
  auto impl = batch.Impl();
  common::CompressedIterator<uint32_t> iterator(
      impl->gidx_buffer.HostVector().data(), impl->NumSymbols());

  for (auto i = 0ull; i < kRows; i++) {
    for (auto j = 0ull; j < columns.size(); j++) {
      if (j == 0) {
        EXPECT_EQ(iterator[i * 2 + j], impl->Cuts().SearchBin(d_data_0[i], j));
      } else {
        EXPECT_EQ(iterator[i * 2 + j], impl->Cuts().SearchBin(d_data_1[i], j));
      }
    }
  }
  EXPECT_EQ(dmat.Info().num_col_, 2);
  EXPECT_EQ(dmat.Info().num_row_, kRows);
  EXPECT_EQ(dmat.Info().num_nonzero_, kRows*2);

}

// Test equivalence with simple DMatrix
TEST(DeviceDMatrix, Equivalent) {
  int bin_sizes[] = {2, 16, 256, 512};
  int sizes[] = {100, 1000, 1500};
  int num_columns = 5;
  for (auto num_rows : sizes) {
    auto x = common::GenerateRandom(num_rows, num_columns);
    for (auto num_bins : bin_sizes) {
      auto dmat = common::GetDMatrixFromData(x, num_rows, num_columns);
      auto x_device = thrust::device_vector<float>(x);
      auto adapter = common::AdapterFromData(x_device, num_rows, num_columns);
      data::DeviceDMatrix device_dmat(
          &adapter, std::numeric_limits<float>::quiet_NaN(), 1, num_bins);

      const auto &batch = *dmat->GetBatches<EllpackPage>({0, num_bins}).begin();
      const auto &device_dmat_batch =
          *device_dmat.GetBatches<EllpackPage>({0, num_bins}).begin();

      ASSERT_EQ(batch.Impl()->Cuts().Values(), device_dmat_batch.Impl()->Cuts().Values());
      ASSERT_EQ(batch.Impl()->gidx_buffer.HostVector(),
                device_dmat_batch.Impl()->gidx_buffer.HostVector());
    }
  }
}

TEST(DeviceDMatrix, IsDense) {
  int num_bins = 16;
  auto test = [num_bins] (float sparsity) {
    HostDeviceVector<float> data;
    std::string interface_str = RandomDataGenerator{10, 10, sparsity}
      .Device(0).GenerateArrayInterface(&data);
    data::CupyAdapter x{interface_str};
    std::unique_ptr<data::DeviceDMatrix> device_dmat{ new data::DeviceDMatrix(
        &x, std::numeric_limits<float>::quiet_NaN(), 1, num_bins) };
    if (sparsity == 0.0) {
      ASSERT_TRUE(device_dmat->IsDense()) << sparsity;
    } else {
      ASSERT_FALSE(device_dmat->IsDense());
    }
  };
  test(0.0);
  test(0.1);
}
