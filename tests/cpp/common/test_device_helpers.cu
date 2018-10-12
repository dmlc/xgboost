/*!
 * Copyright 2017 XGBoost contributors
 */
#include <thrust/device_vector.h>
#include <xgboost/base.h>
#include "../../../src/common/device_helpers.cuh"
#include "../../../src/common/monitor.h"
#include "gtest/gtest.h"

struct Shard { int id; };

TEST(DeviceHelpers, Basic) {
  std::vector<Shard> shards (4);
  for (int i = 0; i < 4; ++i) {
    shards[i].id = i;
  }
  int sum = dh::ReduceShards<int>(&shards, [](Shard& s) { return s.id ; });
  ASSERT_EQ(sum, 6);
}

void CreateTestData(xgboost::bst_uint num_rows, int max_row_size,
                    thrust::host_vector<int> *row_ptr,
                    thrust::host_vector<xgboost::bst_uint> *rows) {
  row_ptr->resize(num_rows + 1);
  int sum = 0;
  for (xgboost::bst_uint i = 0; i <= num_rows; i++) {
    (*row_ptr)[i] = sum;
    sum += rand() % max_row_size;  // NOLINT

    if (i < num_rows) {
      for (int j = (*row_ptr)[i]; j < sum; j++) {
        (*rows).push_back(i);
      }
    }
  }
}

void TestLbs() {
  srand(17);
  dh::CubMemory temp_memory;

  std::vector<int> test_rows = {4, 100, 1000};
  std::vector<int> test_max_row_sizes = {4, 100, 1300};

  for (auto num_rows : test_rows) {
    for (auto max_row_size : test_max_row_sizes) {
      thrust::host_vector<int> h_row_ptr;
      thrust::host_vector<xgboost::bst_uint> h_rows;
      CreateTestData(num_rows, max_row_size, &h_row_ptr, &h_rows);
      dh::DeviceVector<size_t> row_ptr = h_row_ptr;
      dh::DeviceVector<int> output_row(h_rows.size());
      auto d_output_row = output_row.data();

      dh::TransformLbs(0, &temp_memory, h_rows.size(), dh::Raw(row_ptr),
                       row_ptr.size() - 1, false,
                       [=] __device__(size_t idx, size_t ridx) {
                         d_output_row[idx] = ridx;
                       });

      dh::safe_cuda(cudaDeviceSynchronize());
      ASSERT_TRUE(h_rows == output_row);
    }
  }
}

TEST(cub_lbs, Test) { TestLbs(); }

TEST(sumReduce, Test) {
  dh::DeviceVector<float> data(100, 1.0f);
  dh::CubMemory temp;
  auto sum = dh::SumReduction(temp, dh::Raw(data), data.size());
  ASSERT_NEAR(sum, 100.0f, 1e-5);
}

TEST(DeviceHelper, DeviceAllocator) {
  using DeviceMemoryStat = xgboost::common::DeviceMemoryStat;
  DeviceMemoryStat::Ins().Reset();
  DeviceMemoryStat::Ins().SetProfiling(true);

  std::vector<dh::ProfilingDeviceAllocator<float>> allocators(2);
  auto ptr = allocators[0].allocate(12);
  allocators[1] = allocators[0];
  auto usage = DeviceMemoryStat::Ins().GetPtrUsage(&allocators[1]);

  ASSERT_EQ(usage.GetPeak(), sizeof(float)*12);
  ASSERT_EQ(usage.GetAllocCount(), 1);
  ASSERT_EQ(usage.GetRunningSum(), sizeof(float)*12);

  // allocators[0].this is replaced.
  EXPECT_ANY_THROW(allocators[0].deallocate(ptr, 12));

  DeviceMemoryStat::Ins().SetProfiling(false);
}