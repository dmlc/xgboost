/*!
 * Copyright 2017 XGBoost contributors
 */
#include <thrust/device_vector.h>
#include <xgboost/base.h>
#include "../../../src/common/device_helpers.cuh"
#include "../../../src/common/timer.h"
#include "gtest/gtest.h"

using xgboost::common::Span;

struct Shard { int id; };

TEST(DeviceHelpers, ReduceShards) {
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
      thrust::device_vector<size_t> row_ptr = h_row_ptr;
      thrust::device_vector<int> output_row(h_rows.size());
      auto d_output_row = dh::ToSpan(output_row);
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

TEST(DeviceHelpers, CubLbs) { TestLbs(); }

TEST(DeviceHelpers, SumReduce) {
  thrust::device_vector<float> data(100, 1.0f);
  dh::CubMemory temp;
  auto sum = dh::SumReduction(temp, dh::Raw(data), data.size());
  ASSERT_NEAR(sum, 100.0f, 1e-5);
}

void TestAllocator() {
  int n = 10;
  Span<float> a;
  Span<int> b;
  Span<size_t> c;
  dh::BulkAllocator ba;
  ba.Allocate(0, &a, n, &b, n, &c, n);

  // Should be no illegal memory accesses
  dh::LaunchN(0, n, [=] __device__(size_t idx) { c[idx] = a[idx] + b[idx]; });

  dh::safe_cuda(cudaDeviceSynchronize());
}

// Define the test in a function so we can use device lambda
TEST(bulkAllocator, Test) { TestAllocator(); }

#ifdef XGBOOST_USE_NCCL
TEST(DeviceHelpers, MGPU_ExecutePerDevice) {
  xgboost::GPUSet devices = xgboost::GPUSet::AllVisible();
  xgboost::common::Timer timer;
  timer.Start();
  dh::ExecutePerDevice(devices.Size(), [&](int i) {
      dh::safe_cuda(cudaSetDevice(i));
      sleep(1);
    });
  timer.Stop();
  auto elapsed = timer.ElapsedSeconds();
  if (devices.Size() > 1) {
    // Make sure the function is executed in parallel.
    ASSERT_LT(elapsed, devices.Size());
  }
}
#endif  // XGBOOST_USE_NCCL