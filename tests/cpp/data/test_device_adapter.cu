/**
 * Copyright 2019-2024, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <xgboost/data.h>

#include <cstdint>
#include <vector>

#include "../../../src/data/adapter.h"
#include "../../../src/data/device_adapter.cuh"
#include "../helpers.h"
#include "test_array_interface.h"
using namespace xgboost;  // NOLINT

void TestCudfAdapter() {
  constexpr size_t kRowsA{16};
  constexpr size_t kRowsB{16};
  std::vector<Json> columns;
  thrust::device_vector<double> d_data_0(kRowsA);
  thrust::device_vector<uint32_t> d_data_1(kRowsB);

  columns.emplace_back(GenerateDenseColumn<double>("<f8", kRowsA, &d_data_0));
  columns.emplace_back(GenerateDenseColumn<uint32_t>("<u4", kRowsB, &d_data_1));

  Json column_arr{columns};

  std::string str;
  Json::Dump(column_arr, &str);

  data::CudfAdapter adapter(str);

  adapter.Next();
  auto& batch = adapter.Value();
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

TEST(DeviceAdapter, CudfAdapter) { TestCudfAdapter(); }

TEST(DeviceAdapter, EmptyCategories) {
  thrust::device_vector<std::int32_t> names;
  thrust::device_vector<std::int8_t> codes;
  auto j_names = GenerateDenseColumn<std::int32_t>("<i4", 0, &names);
  auto j_codes = GenerateDenseColumn<std::int8_t>("<i1", 1, &codes);

  Json column{Array(std::vector<Json>{j_names, j_codes})};
  Json dataframe{Array(std::vector<Json>{column})};
  auto str = Json::Dump(dataframe);
  EXPECT_THAT([&] { data::CudfAdapter{str}; },
              GMockThrow("Categorical feature must have at least one category."));
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
