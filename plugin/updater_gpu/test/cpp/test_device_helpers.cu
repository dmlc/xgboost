
/*!
 * Copyright 2017 XGBoost contributors
 */
#include <thrust/device_vector.h>
#include <xgboost/base.h>
#include "../../src/device_helpers.cuh"
#include "gtest/gtest.h"

static const std::vector<int> gidx = {0, 2, 5, 1, 3, 6, 0, 2, 0, 7};
static const std::vector<int> row_ptr = {0, 3, 6, 8, 10};
static const std::vector<int> lbs_seg_output = {0, 0, 0, 1, 1, 1, 2, 2, 3, 3};

thrust::device_vector<int> test_lbs() {
  thrust::device_vector<int> device_gidx = gidx;
  thrust::device_vector<int> device_row_ptr = row_ptr;
  thrust::device_vector<int> device_output_row(gidx.size(), 0);
  auto d_output_row = device_output_row.data();
  dh::CubMemory temp_memory;
  dh::TransformLbs(
      0, &temp_memory, gidx.size(), device_row_ptr.data(), row_ptr.size() - 1,
      [=] __device__(int idx, int ridx) { d_output_row[idx] = ridx; });

  dh::safe_cuda(cudaDeviceSynchronize());
  return device_output_row;
}

TEST(lbs, Test) { ASSERT_TRUE(test_lbs() == lbs_seg_output); }
