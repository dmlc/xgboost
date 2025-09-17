/**
 * Copyright 2019-2024, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/data.h>
#include "../../../src/data/adapter.h"
#include "../helpers.h"
#include <thrust/device_vector.h>
#include "../../../src/data/device_adapter.cuh"
#include "test_array_interface.h"
using namespace xgboost;  // NOLINT

void TestCudfAdapter()
{
  constexpr size_t kRowsA {16};
  constexpr size_t kRowsB {16};
  std::vector<Json> columns;
  thrust::device_vector<double> d_data_0(kRowsA);
  thrust::device_vector<uint32_t> d_data_1(kRowsB);

  columns.emplace_back(GenerateDenseColumn<double>("<f8", kRowsA, &d_data_0));
  columns.emplace_back(GenerateDenseColumn<uint32_t>("<u4", kRowsB, &d_data_1));

  Json column_arr {columns};

  std::string str;
  Json::Dump(column_arr, &str);

  data::CudfAdapter adapter(str);

  adapter.Next();
  auto & batch = adapter.Value();
  EXPECT_EQ(batch.Size(), kRowsA + kRowsB);

  EXPECT_NO_THROW({
    dh::LaunchN(batch.Size(), [=] __device__(size_t idx) {
      auto element = batch.GetElement(idx);
      KERNEL_CHECK(element.row_idx == idx / 2);
      if (idx % 2 == 0) {
        KERNEL_CHECK(element.column_idx == 0);
        KERNEL_CHECK(element.value == element.row_idx * 2.0f);
      } else {
        KERNEL_CHECK(element.column_idx == 1);
        KERNEL_CHECK(element.value == element.row_idx * 2.0f);
      }
    });
    dh::safe_cuda(cudaDeviceSynchronize());
  });
}

TEST(DeviceAdapter, CudfAdapter) {
  TestCudfAdapter();
}

namespace xgboost::data {
TEST(DeviceAdapter, GetRowCounts) {
  auto ctx = MakeCUDACtx(0);

  for (bst_feature_t n_features : {1, 2, 4, 64, 128, 256}) {
    HostDeviceVector<float> storage;
    auto str_arr = RandomDataGenerator{8192, n_features, 0.0}
                       .Device(ctx.Device())
                       .GenerateArrayInterface(&storage);
    auto adapter = CupyAdapter{str_arr};
    HostDeviceVector<bst_idx_t> offset(adapter.NumRows() + 1, 0);
    offset.SetDevice(ctx.Device());
    auto rstride = GetRowCounts(&ctx, adapter.Value(), offset.DeviceSpan(), ctx.Device(),
                                std::numeric_limits<float>::quiet_NaN());
    ASSERT_EQ(rstride, n_features);
  }
}
}  // namespace xgboost::data
