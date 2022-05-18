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
    if (ridx > 4) {
      return 1;
    }
    else {
      return 2;
    }
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
    if (ridx < 7) {
      return 3;
    }
    return 4;
  });
  EXPECT_EQ(rp.GetRows(3).size(), 2);
  EXPECT_EQ(rp.GetRows(4).size(), 3);
}

TEST(RowPartitioner, Batch) { TestUpdatePositionBatch(); }

void TestFinalise() {
  const int kNumRows = 10;

  ObjInfo task{ObjInfo::kRegression, false, false};
  HostDeviceVector<bst_node_t> position;
  Context ctx;
  ctx.gpu_id = 0;

  {
    RowPartitioner rp(0, kNumRows);
    rp.FinalisePosition(
        &ctx, task, &position,
        [=] __device__(RowPartitioner::RowIndexT ridx, int position) { return 7; },
        [] XGBOOST_DEVICE(size_t idx) { return false; });

    auto position = rp.GetPositionHost();
    for (auto p : position) {
      EXPECT_EQ(p, 7);
    }
  }

  /**
   * Test for sampling.
   */
  dh::device_vector<float> hess(kNumRows);
  for (size_t i = 0; i < hess.size(); ++i) {
    // removed rows, 0, 3, 6, 9
    if (i % 3 == 0) {
      hess[i] = 0;
    } else {
      hess[i] = i;
    }
  }

  auto d_hess = dh::ToSpan(hess);

  RowPartitioner rp(0, kNumRows);
  rp.FinalisePosition(
      &ctx, task, &position,
      [] __device__(RowPartitioner::RowIndexT ridx, bst_node_t position) {
        return ridx % 2 == 0 ? 1 : 2;
      },
      [d_hess] __device__(size_t ridx) { return d_hess[ridx] - 0.f == 0.f; });

  auto const& h_position = position.ConstHostVector();
  for (size_t ridx = 0; ridx < h_position.size(); ++ridx) {
    if (ridx % 3 == 0) {
      ASSERT_LT(h_position[ridx], 0);
    } else {
      ASSERT_EQ(h_position[ridx], ridx % 2 == 0 ? 1 : 2);
    }
  }
}

TEST(RowPartitioner, Finalise) { TestFinalise(); }


const int kMaxBatch = 32;
template <typename OpDataT>
struct KernelArgs {
  Segment segments[kMaxBatch];
  OpDataT data[kMaxBatch];

  // Given a global thread idx, assign it to an item from one of the segments
  __device__ void AssignBatch(std::size_t idx, int &batch_idx, std::size_t &item_idx) const {
    std::size_t sum = 0;
    for (int i = 0; i < kMaxBatch; i++) {
      if (sum + segments[i].Size() > idx) {
        batch_idx = i;
        item_idx = (idx - sum) + segments[i].begin;
        break;
      }
      sum += segments[i].Size();
    }
  }
  std::size_t TotalRows() const {
    std::size_t total_rows = 0;
    for (auto segment : segments) {
      total_rows += segment.Size();
    }
    return total_rows;
  }
};

template <typename RowIndexT, typename OpT, typename OpDataT>
void GetLeftCounts(const KernelArgs<OpDataT>&args,common::Span<RowIndexT> ridx,
                   common::Span<unsigned long long int> d_left_counts, OpT op
                   ) {

  // Launch 1 thread for each row
  dh::LaunchN<1, 128>(args.TotalRows(), [=] __device__(std::size_t idx) {
    // Assign this thread to a row
    int batch_idx;
    std::size_t item_idx;
    args.AssignBatch(idx, batch_idx, item_idx);
    auto op_res = op(ridx[item_idx], args.data[batch_idx]);
    atomicAdd(&d_left_counts[batch_idx], op(ridx[item_idx], args.data[batch_idx]));
  });
}

struct IndexFlagTuple {
  size_t idx;
  bool flag;
  size_t flag_scan;
  int batch_idx;
};

struct IndexFlagOp {
  __device__ IndexFlagTuple operator()(const IndexFlagTuple& a, const IndexFlagTuple& b) const {
    if (a.batch_idx == b.batch_idx) {
      return {b.idx, b.flag, a.flag_scan + b.flag_scan, b.batch_idx};
    } else {
      return b;
    }
  }
};

template<typename OpDataT,typename OpT>
struct WriteResultsFunctor {
  KernelArgs<OpDataT> args;
  OpT op;
  common::Span<RowPartitioner::RowIndexT> ridx_in;
  common::Span<RowPartitioner::RowIndexT> ridx_out;
  common::Span<unsigned long long int> left_counts;

  __device__ IndexFlagTuple operator()(const IndexFlagTuple& x) {
    // the ex_scan_result represents how many rows have been assigned to left
    // node so far during scan.
    std::size_t scatter_address;
    if (x.flag) {
      scatter_address = args.segments[x.batch_idx].begin + x.flag_scan - 1;  // -1 because inclusive scan
    } else {
      // current number of rows belong to right node + total number of rows
      // belong to left node
      scatter_address  = (x.idx - x.flag_scan) + left_counts[x.batch_idx];
    }
    ridx_out[scatter_address] = ridx_in[x.idx];
    // Discard
    return {};
  }
};

template <typename RowIndexT, typename OpT, typename OpDataT>
void SortPositionBatch(const KernelArgs<OpDataT>& args, common::Span<RowIndexT> ridx,
                       common::Span<RowIndexT> ridx_tmp,
                       common::Span<unsigned long long int> left_counts, OpT op,
                       cudaStream_t stream) {
  WriteResultsFunctor<OpDataT,OpT> write_results{args,op,ridx, ridx_tmp, left_counts};
  auto discard_write_iterator =
      thrust::make_transform_output_iterator(dh::TypedDiscard<IndexFlagTuple>(), write_results);
  auto counting = thrust::make_counting_iterator(0llu);
  auto input_iterator =
      dh::MakeTransformIterator<IndexFlagTuple>(counting, [=] __device__(size_t idx) {
        int batch_idx;
        std::size_t item_idx;
        args.AssignBatch(idx, batch_idx, item_idx);
        auto go_left = op(ridx[item_idx], args.data[batch_idx]);
        return IndexFlagTuple{item_idx, go_left,go_left, batch_idx};
      });
  size_t temp_bytes = 0;
  cub::DeviceScan::InclusiveScan(nullptr, temp_bytes, input_iterator,
                                 discard_write_iterator, IndexFlagOp(),
                                 args.TotalRows(), stream);
  dh::TemporaryArray<int8_t> temp(temp_bytes);
  cub::DeviceScan::InclusiveScan(temp.data().get(), temp_bytes, input_iterator,
                                 discard_write_iterator, IndexFlagOp(), args.TotalRows(), stream);

  // copy active segments back to original buffer
  dh::LaunchN(args.TotalRows(), [=] __device__(std::size_t idx) {
    // Assign this thread to a row
    int batch_idx;
    std::size_t item_idx;
    args.AssignBatch(idx, batch_idx, item_idx);
    ridx[item_idx] = ridx_tmp[item_idx];
  });
}

void TestSortPositionBatch(const std::vector<int>& ridx_in, const std::vector<Segment>& segments) {
  thrust::device_vector<uint32_t> ridx = ridx_in;
  thrust::device_vector<uint32_t> ridx_tmp(ridx_in.size());
  thrust::device_vector<unsigned long long int> left_counts(segments.size());

  auto op = [=] __device__(auto ridx, int data) { return ridx % 2 == 0; };
  std::vector<int> op_data(segments.size());
  KernelArgs<int> args;
  std::copy(segments.begin(), segments.end(), args.segments);
  std::copy(op_data.begin(), op_data.end(), args.data);
  GetLeftCounts(args, dh::ToSpan(ridx), dh::ToSpan(left_counts), op);
  SortPositionBatch(args, dh::ToSpan(ridx), dh::ToSpan(ridx_tmp), dh::ToSpan(left_counts), op,
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

}  // namespace tree
}  // namespace xgboost
