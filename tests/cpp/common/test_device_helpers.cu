
/*!
 * Copyright 2017 XGBoost contributors
 */
#include <thrust/device_vector.h>
#include <xgboost/base.h>
#include "../../../src/common/device_helpers.cuh"
#include "../../../src/common/timer.h"
#include "gtest/gtest.h"

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
  thrust::device_vector<float> data(100, 1.0f);
  dh::CubMemory temp;
  auto sum = dh::SumReduction(temp, dh::Raw(data), data.size());
  ASSERT_NEAR(sum, 100.0f, 1e-5);
}

TEST(BulkAllocator, Independent) {
  dh::BulkAllocatorTemp ba;
  auto devices = xgboost::GPUSet::AllVisible();
  if (devices.IsEmpty()) {
    LOG(WARNING) << "Empty devices.";
    return;
  }
  dh::DSpan<int> span_i;
  dh::DSpan<float> span_f;
  dh::DSpan<double> span_d;
  ba.Allocate(0, &span_i, 16, &span_f, 32, &span_d, 64);

  ASSERT_TRUE(span_i.data());
  ASSERT_EQ(span_i.size(), 16);

  ASSERT_TRUE(span_f.data());
  ASSERT_EQ(span_f.size(), 32);

  ASSERT_TRUE(span_d.data());
  ASSERT_EQ(span_d.size(), 64);

  dh::DoubleBuffer<int> buf_i;
  dh::DoubleBuffer<float> buf_f;
  dh::DoubleBuffer<double> buf_d;

  ba.Allocate(0, &buf_i, 16, &buf_f, 32, &buf_d, 64);

  ASSERT_TRUE(buf_i.S1().data());
  ASSERT_EQ(buf_i.S1().size(), 16);
  ASSERT_EQ(buf_i.S2().size(), 16);

  ASSERT_TRUE(buf_f.S1().data());
  ASSERT_EQ(buf_f.S1().size(), 32);
  ASSERT_EQ(buf_f.S2().size(), 32);

  ASSERT_TRUE(buf_d.S1().data());
  ASSERT_EQ(buf_d.S1().size(), 64);
  ASSERT_EQ(buf_d.S2().size(), 64);
}

TEST(BulkAllocator, Mixed){
  dh::BulkAllocatorTemp ba;
  auto devices = xgboost::GPUSet::AllVisible();
  if (devices.IsEmpty()) {
    LOG(WARNING) << "Empty devices.";
    return;
  }

  dh::DSpan<float> span_f;
  dh::DSpan<double> span_d;
  ba.Allocate(0, &span_f, 32, &span_d, 64);

  dh::DoubleBuffer<float> buf_f;
  dh::DoubleBuffer<double> buf_d;
  ba.Allocate(0, &span_f, 16, &span_d, 32, &buf_f, 64, &buf_d, 128);

  ASSERT_EQ(span_f.size(), 16);
  ASSERT_EQ(span_d.size(), 32);
  ASSERT_EQ(buf_f.Size(), 64);
  ASSERT_EQ(buf_d.Size(), 128);
}