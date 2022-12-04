/*!
 * Copyright 2019-2022 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <vector>

#include "../../../../src/tree/gpu_hist/row_partitioner.cuh"
#include "../../helpers.h"
#include "xgboost/base.h"
#include "xgboost/context.h"
#include "xgboost/task.h"
#include "xgboost/tree_model.h"

namespace xgboost {
namespace tree {

void TestUpdatePositionBatch() {
  const int kNumRows = 10;
  RowPartitioner rp(0, kNumRows);
  auto rows = rp.GetRowsHost(0);
  EXPECT_EQ(rows.size(), kNumRows);
  for (auto i = 0ull; i < kNumRows; i++) {
    EXPECT_EQ(rows[i], i);
  }
  std::vector<int> extra_data = {0};
  // Send the first five training instances to the right node
  // and the second 5 to the left node
  rp.UpdatePositionBatch({0}, {1}, {2}, extra_data, [=] __device__(RowPartitioner::RowIndexT ridx, int) {
    return ridx > 4;
  });
  rows = rp.GetRowsHost(1);
  for (auto r : rows) {
    EXPECT_GT(r, 4);
  }
  rows = rp.GetRowsHost(2);
  for (auto r : rows) {
    EXPECT_LT(r, 5);
  }

  // Split the left node again
  rp.UpdatePositionBatch({1}, {3}, {4}, extra_data,[=] __device__(RowPartitioner::RowIndexT ridx, int) {
    return ridx < 7;
  });
  EXPECT_EQ(rp.GetRows(3).size(), 2);
  EXPECT_EQ(rp.GetRows(4).size(), 3);
}

TEST(RowPartitioner, Batch) { TestUpdatePositionBatch(); }

void TestSortPositionBatch(const std::vector<int>& ridx_in, const std::vector<Segment>& segments) {
  thrust::device_vector<uint32_t> ridx = ridx_in;
  thrust::device_vector<uint32_t> ridx_tmp(ridx_in.size());
  thrust::device_vector<bst_uint> counts(segments.size());

  auto op = [=] __device__(auto ridx, int data) { return ridx % 2 == 0; };
  std::vector<int> op_data(segments.size());
  std::vector<PerNodeData<int>> h_batch_info(segments.size());
  dh::TemporaryArray<PerNodeData<int>> d_batch_info(segments.size());

  std::size_t total_rows = 0;
  for (size_t i = 0; i < segments.size(); i++) {
    h_batch_info[i] = {segments.at(i), 0};
    total_rows += segments.at(i).Size();
  }
  dh::safe_cuda(cudaMemcpyAsync(d_batch_info.data().get(), h_batch_info.data(),
                                h_batch_info.size() * sizeof(PerNodeData<int>), cudaMemcpyDefault,
                                nullptr));
  dh::device_vector<int8_t> tmp;
  SortPositionBatch<uint32_t, decltype(op), int>(dh::ToSpan(d_batch_info), dh::ToSpan(ridx),
                                                 dh::ToSpan(ridx_tmp), dh::ToSpan(counts),
                                                 total_rows, op, &tmp, nullptr);

  auto op_without_data = [=] __device__(auto ridx) { return ridx % 2 == 0; };
  for (size_t i = 0; i < segments.size(); i++) {
    auto begin = ridx.begin() + segments[i].begin;
    auto end = ridx.begin() + segments[i].end;
    bst_uint count = counts[i];
    auto left_partition_count =
        thrust::count_if(thrust::device, begin, begin + count, op_without_data);
    EXPECT_EQ(left_partition_count, count);
    auto right_partition_count =
        thrust::count_if(thrust::device, begin + count, end, op_without_data);
    EXPECT_EQ(right_partition_count, 0);
  }
}

TEST(GpuHist, SortPositionBatch) { 
  TestSortPositionBatch({0, 1, 2, 3, 4, 5}, {{0, 3}, {3, 6}}); 
  TestSortPositionBatch({0, 1, 2, 3, 4, 5}, {{0, 1}, {3, 6}}); 
  TestSortPositionBatch({0, 1, 2, 3, 4, 5}, {{0, 6}});
  TestSortPositionBatch({0, 1, 2, 3, 4, 5}, {{3, 6}, {0, 2}});
}

}  // namespace tree
}  // namespace xgboost
