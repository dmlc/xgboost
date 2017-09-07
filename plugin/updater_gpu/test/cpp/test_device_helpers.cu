
/*!
 * Copyright 2017 XGBoost contributors
 */
#include <thrust/device_vector.h>
#include <xgboost/base.h>
#include "../../src/device_helpers.cuh"
#include "gtest/gtest.h"

void CreateTestData(xgboost::bst_uint num_rows, int max_row_size,
                    thrust::host_vector<int> *row_ptr,
                    thrust::host_vector<xgboost::bst_uint> *rows) {
  row_ptr->resize(num_rows + 1);
  int sum = 0;
  for (int i = 0; i <= num_rows; i++) {
    (*row_ptr)[i] = sum;
    sum += rand() % max_row_size;  // NOLINT

    if (i < num_rows) {
      for (int j = (*row_ptr)[i]; j < sum; j++) {
        (*rows).push_back(i);
      }
    }
  }
}

void SpeedTest() {
  int num_rows = 1000000;
  int max_row_size = 100;
  dh::CubMemory temp_memory;
  thrust::host_vector<int> h_row_ptr;
  thrust::host_vector<xgboost::bst_uint> h_rows;
  CreateTestData(num_rows, max_row_size, &h_row_ptr, &h_rows);
  thrust::device_vector<int> row_ptr = h_row_ptr;
  thrust::device_vector<int> output_row(h_rows.size());
  auto d_output_row = output_row.data();

  dh::Timer t;
  dh::TransformLbs(
      0, &temp_memory, h_rows.size(), dh::raw(row_ptr), row_ptr.size() - 1, false,
      [=] __device__(size_t idx, size_t ridx) { d_output_row[idx] = ridx; });

  dh::safe_cuda(cudaDeviceSynchronize());
  double time = t.elapsedSeconds();
  const int mb_size = 1048576;
  size_t size = (sizeof(int) * h_rows.size()) / mb_size;
  printf("size: %llumb, time: %fs, bandwidth: %fmb/s\n", size, time,
         size / time);
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
      auto d_output_row = output_row.data();

      dh::TransformLbs(0, &temp_memory, h_rows.size(), dh::raw(row_ptr),
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
