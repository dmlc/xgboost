/*!
 * Copyright 2019-2022 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <algorithm>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include "../../../../src/tree/gpu_hist/row_partitioner.cuh"
#include "../../helpers.h"
#include "xgboost/base.h"
#include "xgboost/generic_parameters.h"
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
  thrust::device_vector<unsigned long long int> left_counts(segments.size());
  thrust::device_vector<IndexFlagTuple> scan_tmp(ridx_in.size());

  auto op = [=] __device__(auto ridx, int data) { return ridx % 2 == 0; };
  std::vector<int> op_data(segments.size());
  KernelBatchArgs<int> args;
  std::copy(segments.begin(), segments.end(), args.segments);
  std::copy(op_data.begin(), op_data.end(), args.data);
  GetLeftCounts(args, dh::ToSpan(ridx), dh::ToSpan(scan_tmp),dh::ToSpan(left_counts), op);
  SortPositionBatch(args, dh::ToSpan(ridx), dh::ToSpan(ridx_tmp), dh::ToSpan(scan_tmp), dh::ToSpan(left_counts),
                    nullptr);

  auto op_without_data = [=] __device__(auto ridx) { return ridx % 2 == 0; };
  for (int i = 0; i < segments.size(); i++) {
    auto begin = ridx.begin() + segments[i].begin;
    auto end = ridx.begin() + segments[i].end;
    auto left_partition_count =
        thrust::count_if(thrust::device, begin, begin + left_counts[i], op_without_data);
    EXPECT_EQ(left_partition_count, left_counts[i]);
    auto right_partition_count =
        thrust::count_if(thrust::device, begin + left_counts[i], end, op_without_data);
    EXPECT_EQ(right_partition_count, 0);
  }
}

TEST(GpuHist, SortPositionBatch) { 
  TestSortPositionBatch({0, 1, 2, 3, 4, 5}, {{0, 3}, {3, 6}}); 
  TestSortPositionBatch({0, 1, 2, 3, 4, 5}, {{0, 1}, {3, 6}}); 
  TestSortPositionBatch({0, 1, 2, 3, 4, 5}, {{0, 6}});
  TestSortPositionBatch({0, 1, 2, 3, 4, 5}, {{3, 6}, {0, 2}});
}

void TestAtomicIncrement(const std::vector<int>& group_in, const std::vector<int>& increment_in) {
  thrust::device_vector<int> group(group_in);
  thrust::device_vector<int> increment(increment_in);
  thrust::device_vector<unsigned long long> reference(group_in.size());
  thrust::device_vector<unsigned long long> result(group_in.size());

  auto d_group = group.data().get();
  auto d_increment = increment.data().get();
  auto d_reference = reference.data().get();
  auto d_result = result.data().get();
  dh::LaunchN(group.size(), [=] __device__(std::size_t idx) {
    AtomicIncrement(d_result, d_increment[idx], d_group[idx]);
    atomicAdd(d_reference + d_group[idx], d_increment[idx]);
  });

  EXPECT_EQ(reference, result);
}

TEST(GpuHist, AtomicIncrement) {
  TestAtomicIncrement({0, 0, 0}, {1, 0, 1});
  TestAtomicIncrement({0, 0, 1}, {1, 0, 1});
}
}  // namespace tree
}  // namespace xgboost
