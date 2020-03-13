
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
using namespace xgboost;  // NOLINT

TEST(DeviceDMatrix, Simple) {
  int num_rows = 1000;
  int num_columns = 50;
  auto x = common::GenerateRandom(num_rows, num_columns);
  auto x_device = thrust::device_vector<float>(x);
  auto adapter = common::AdapterFromData(x_device, num_rows, num_columns);

  data::DeviceDMatrix device_dmat(&adapter,
                                  std::numeric_limits<float>::quiet_NaN(), 1);

  auto &batch = *device_dmat.GetBatches<EllpackPage>({0, 256, 0}).begin();
  auto impl = batch.Impl();
  common::CompressedIterator<uint32_t> iterator(
      impl->gidx_buffer.HostVector().data(), impl->NumSymbols());
  for(auto i = 0ull; i < x.size(); i++)
  {
    int column_idx = i % num_columns;
    EXPECT_EQ(impl->cuts_.SearchBin(x[i], column_idx), iterator[i]);
  }

}

TEST(DeviceDMatrix, Missing) {
 const  float kMissing=std::numeric_limits<float>::quiet_NaN();
  int num_rows = 10;
  int num_columns = 2;
  auto x = common::GenerateRandom(num_rows, num_columns);
  x[1] = kMissing;
  x[5] = kMissing;
  x[6] = kMissing;
  auto x_device = thrust::device_vector<float>(x);
  auto adapter = common::AdapterFromData(x_device, num_rows, num_columns);

  data::DeviceDMatrix device_dmat(&adapter, kMissing, 1);

  auto &batch = *device_dmat.GetBatches<EllpackPage>({0, 256, 0}).begin();
  auto impl = batch.Impl();
  common::CompressedIterator<uint32_t> iterator(
      impl->gidx_buffer.HostVector().data(), impl->NumSymbols());
  EXPECT_EQ(iterator[1], impl->GetDeviceAccessor(0).NullValue());
  EXPECT_EQ(iterator[5], impl->GetDeviceAccessor(0).NullValue());
  EXPECT_EQ(iterator[7], impl->GetDeviceAccessor(0).NullValue()); // null values get placed after

}
