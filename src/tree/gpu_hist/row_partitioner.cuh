/*!
 * Copyright 2017-2022 XGBoost contributors
 */
#pragma once
#include <limits>
#include <vector>
#include "xgboost/base.h"
#include "../../common/device_helpers.cuh"
#include "xgboost/generic_parameters.h"
#include "xgboost/task.h"
#include "xgboost/tree_model.h"
#include <thrust/execution_policy.h>

namespace xgboost {
namespace tree {

  /** \brief Used to demarcate a contiguous set of row indices associated with
 * some tree node. */
struct Segment {
  size_t begin{0};
  size_t end{0};

  Segment() = default;

  Segment(size_t begin, size_t end) : begin(begin), end(end) { CHECK_GE(end, begin); }
  __host__ __device__ size_t Size() const { return end - begin; }
};

using PartitionCountsT = thrust::pair<bst_uint,bst_uint>;

template <typename OpDataT>
struct KernelMemcpyArgs {
  Segment segment;
  OpDataT data;
};

template <typename OpDataT>
__device__ void AssignBatch(const common::Span<KernelMemcpyArgs<OpDataT>> batch_info,
                            std::size_t idx, int16_t& batch_idx, std::size_t& item_idx, OpDataT&data) {
  const auto ptr = batch_info.data();
  std::size_t sum = 0;

  for (int16_t i = 0; i < batch_info.size(); i++) {
    if (sum + ptr[i].segment.Size() > idx) {
      batch_idx = i;
      item_idx = (idx - sum) + ptr[i].segment.begin;
      data = ptr[i].data;
      break;
    }
    sum += ptr[i].segment.Size();
  }
}

// We can scan over this tuple, where the scan gives us information on how to partition inputs
// according to the flag
struct IndexFlagTuple {
  bst_uint idx;            // The location of the item we are working on in ridx_
  bst_uint flag_scan;      // This gets populated after scanning
  bst_uint segment_start;  // Start offset of this node segment
  bst_uint segment_end;  // End offset of this node segment
  int16_t batch_idx;       // Which node in the batch does this item belong to
  bool flag;               // Result of op (is this item going left?)
};

struct IndexFlagOp {
  __device__ IndexFlagTuple operator()(const IndexFlagTuple& a, const IndexFlagTuple& b) const {
    // Segmented scan - resets if we cross batch boundaries
    if (a.batch_idx == b.batch_idx) {
      // Accumulate the flags, everything else stays the same
      return {b.idx, a.flag_scan + b.flag_scan, b.segment_start, b.segment_end,b.batch_idx, b.flag};
    } else {
      return b;
    }
  }
};


// This is a transformer output iterator
// It takes the result of the scan and performs the partition
// To understand how a scan is used to partition elements see:
// Harris, Mark, Shubhabrata Sengupta, and John D. Owens. "Parallel prefix sum (scan) with CUDA."
// GPU gems 3.39 (2007): 851-876.
struct WriteResultsFunctor {
  bst_uint* ridx_in;
  bst_uint* ridx_out;
  PartitionCountsT *counts;

  __device__ IndexFlagTuple operator()(const IndexFlagTuple& x) {
    std::size_t scatter_address;
    if (x.flag) {
      bst_uint num_previous_flagged = x.flag_scan - 1; // -1 because inclusive scan
      scatter_address = x.segment_start + num_previous_flagged;  
    } else {

      bst_uint num_previous_unflagged = (x.idx - x.segment_start) - x.flag_scan;
      scatter_address = x.segment_end - num_previous_unflagged - 1;
    }
    ridx_out[scatter_address] = ridx_in[x.idx];

    if (x.idx == (x.segment_end - 1)) {
      // Write out counts
      counts[x.batch_idx] = {x.flag_scan,0};
    }

    // Discard
    return {};
  }
};

template <typename RowIndexT, typename OpT, typename OpDataT>
void SortPositionBatch(const common::Span<KernelMemcpyArgs<OpDataT>> batch_info,
                       common::Span<RowIndexT> ridx, common::Span<RowIndexT> ridx_tmp,
                       common::Span<PartitionCountsT> d_counts, std::size_t total_rows,
                       OpT op, cudaStream_t stream) {
  WriteResultsFunctor write_results{ridx.data(), ridx_tmp.data(), d_counts.data()};

  auto discard_write_iterator =
      thrust::make_transform_output_iterator(dh::TypedDiscard<IndexFlagTuple>(), write_results);
  auto counting = thrust::make_counting_iterator(0llu);
  auto input_iterator =
      dh::MakeTransformIterator<IndexFlagTuple>(counting, [=] __device__(size_t idx) {
        int16_t batch_idx;
        std::size_t item_idx;
        OpDataT data;
        AssignBatch(batch_info, idx, batch_idx, item_idx, data);
        auto op_res = op(ridx[item_idx], data);
        return IndexFlagTuple{bst_uint(item_idx),
                              op_res,
                              bst_uint(batch_info.data()[batch_idx].segment.begin),
                              bst_uint(batch_info.data()[batch_idx].segment.end),
                              batch_idx,
                              op_res};
      });
  size_t temp_bytes = 0;
  cub::DeviceScan::InclusiveScan(nullptr, temp_bytes, input_iterator, discard_write_iterator,
                                 IndexFlagOp(), total_rows, stream);
  dh::TemporaryArray<int8_t> temp(temp_bytes);
  cub::DeviceScan::InclusiveScan(temp.data().get(), temp_bytes, input_iterator,
                                 discard_write_iterator, IndexFlagOp(), total_rows, stream);

  // copy active segments back to original buffer
  dh::LaunchN(total_rows, stream, [=] __device__(std::size_t idx) {
    int16_t batch_idx;
    std::size_t item_idx;
    OpDataT data;
    AssignBatch(batch_info, idx, batch_idx, item_idx, data);
    ridx[item_idx] = ridx_tmp[item_idx];
  });
}


__forceinline__ __device__ uint32_t __lanemask_lt() { return ((uint32_t)1 << cub::LaneId()) - 1; }

/*! \brief Count how many rows are assigned to left node. */
__forceinline__ __device__ uint32_t AtomicIncrement(PartitionCountsT* d_counts, bool go_left,
                                                int16_t batch_idx) {
  int mask = __activemask();
  int leader = __ffs(mask) - 1;
  unsigned int prefix = __popc(mask & __lanemask_lt());
  bool group_is_contiguous = __all_sync(mask, batch_idx == __shfl_sync(mask, batch_idx, leader));
  // If all threads here are working on the same node
  // we can do a more efficient reduction with warp intrinsics
  if (group_is_contiguous) {
    unsigned ballot = __ballot_sync(mask, go_left);
    uint32_t global_left_count = 0;
    uint32_t global_right_count = 0;
    if (prefix == 0) {
      global_left_count = atomicAdd(&d_counts->first, __popc(ballot));
      global_right_count = atomicAdd(&d_counts->second, __popc(mask) - __popc(ballot));
    }
    global_left_count = __shfl_sync(mask, global_left_count, leader);
    global_right_count = __shfl_sync(mask, global_right_count, leader);
    uint32_t local_left_count = __popc(ballot & __lanemask_lt());
    uint32_t local_right_count = __popc(mask & __lanemask_lt()) - local_left_count;

    if (go_left) {
      return global_left_count + local_left_count;
    } else {
      return global_right_count + local_right_count;
    }

  } else {
    auto address = go_left ? &d_counts->first : &d_counts->second;
    return atomicAdd(address, 1);
  }
}

template <int kBlockSize, typename RowIndexT, typename OpT, typename OpDataT>
__global__ __launch_bounds__(kBlockSize) void SortPositionBatchUnstableKernel(
    const common::Span<KernelMemcpyArgs<OpDataT>> d_batch_info, common::Span<RowIndexT> d_ridx,
    common::Span<RowIndexT> ridx_tmp, common::Span<PartitionCountsT> counts, OpT op,
    std::size_t total_rows) {
  __shared__ KernelMemcpyArgs<OpDataT> s_batch_info[32];
  for (int i = threadIdx.x; i < d_batch_info.size(); i += kBlockSize) {
    s_batch_info[i] = d_batch_info.data()[i];
  }
  const common::Span<KernelMemcpyArgs<OpDataT>> batch_info(s_batch_info, d_batch_info.size());
  __syncthreads();

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_rows; idx += blockDim.x * gridDim.x) {
    int16_t batch_idx;
    std::size_t item_idx;
    OpDataT data;
    AssignBatch(batch_info, idx, batch_idx, item_idx, data);
    auto segment = batch_info[batch_idx].segment;
    auto ridx = d_ridx[item_idx];
    auto op_res = op(ridx, data);
    auto current_num_items = AtomicIncrement(&counts.data()[batch_idx], op_res, batch_idx);
    auto destination_address =
        op_res ? segment.begin + current_num_items : segment.end - current_num_items - 1;
    ridx_tmp[destination_address] = ridx;
  }
}

template <typename RowIndexT, typename OpT, typename OpDataT>
void SortPositionBatchUnstable(const common::Span<KernelMemcpyArgs<OpDataT>> batch_info,
                               common::Span<RowIndexT> ridx, common::Span<RowIndexT> ridx_tmp,
                               common::Span<PartitionCountsT> d_counts, std::size_t total_rows,
                               OpT op, cudaStream_t stream) {
  CHECK_LE(batch_info.size(), 32);
  constexpr int kBlockSize = 256;
  const int grid_size =
      std::max(256, static_cast<int>(xgboost::common::DivRoundUp(total_rows, kBlockSize)));

  SortPositionBatchUnstableKernel<kBlockSize>
      <<<grid_size, kBlockSize, 0, stream>>>(batch_info, ridx, ridx_tmp, d_counts, op, total_rows);

  // copy active segments back to original buffer
  dh::LaunchN(total_rows, stream, [=] __device__(std::size_t idx) {
    int16_t batch_idx;
    std::size_t item_idx;
    OpDataT data;
    AssignBatch(batch_info, idx, batch_idx, item_idx, data);
    ridx[item_idx] = ridx_tmp[item_idx];
  });
}

/** \brief Class responsible for tracking subsets of rows as we add splits and
 * partition training rows into different leaf nodes. */
class RowPartitioner {
 public:
  using RowIndexT = bst_uint;

 private:
  int device_idx_;
  /*! \brief In here if you want to find the rows belong to a node nid, first you need to
   * get the indices segment from ridx_segments[nid], then get the row index that
   * represents position of row in input data X.  `RowPartitioner::GetRows` would be a
   * good starting place to get a sense what are these vector storing.
   *
   * node id -> segment -> indices of rows belonging to node
   */
  /*! \brief Range of row index for each node, pointers into ridx below. */
  std::vector<Segment> ridx_segments_;
  /*! \brief mapping for node id -> rows.
   * This looks like:
   * node id  |    1    |    2   |
   * rows idx | 3, 5, 1 | 13, 31 |
   */
  dh::TemporaryArray<RowIndexT> ridx_;
  // Staging area for sorting ridx
  dh::TemporaryArray<RowIndexT> ridx_tmp_;
  dh::PinnedMemory pinned_;
  dh::PinnedMemory pinned2_;
  std::vector<cudaStream_t> streams_;

 public:
  RowPartitioner(int device_idx, size_t num_rows);
  ~RowPartitioner();
  RowPartitioner(const RowPartitioner&) = delete;
  RowPartitioner& operator=(const RowPartitioner&) = delete;

  /**
   * \brief Gets the row indices of training instances in a given node.
   */
  common::Span<const RowIndexT> GetRows(bst_node_t nidx);

  /**
   * \brief Gets all training rows in the set.
   */
  common::Span<const RowIndexT> GetRows();

  /**
   * \brief Convenience method for testing
   */
  std::vector<RowIndexT> GetRowsHost(bst_node_t nidx);

  template <typename UpdatePositionOpT, typename OpDataT>
  void UpdatePositionBatch(const std::vector<bst_node_t>& nidx,
                           const std::vector<bst_node_t>& left_nidx,
                           const std::vector<bst_node_t>& right_nidx,
                           const std::vector<OpDataT>& op_data, UpdatePositionOpT op) {
    if (nidx.empty()) return;
    CHECK_EQ(nidx.size(), left_nidx.size());
    CHECK_EQ(nidx.size(), right_nidx.size());
    CHECK_EQ(nidx.size(), op_data.size());

    auto h_batch_info = pinned2_.GetSpan<KernelMemcpyArgs<OpDataT>>(nidx.size());
    dh::TemporaryArray<KernelMemcpyArgs<OpDataT>> d_batch_info(nidx.size());

    std::size_t total_rows = 0;
    for (int i = 0; i < nidx.size(); i++) {
      h_batch_info[i] = {ridx_segments_.at(nidx.at(i)), op_data.at(i)};
      total_rows += ridx_segments_.at(nidx.at(i)).Size();
    }
    dh::safe_cuda(cudaMemcpyAsync(d_batch_info.data().get(), h_batch_info.data(),
                                  h_batch_info.size() * sizeof(KernelMemcpyArgs<OpDataT>),
                                  cudaMemcpyDefault, streams_[1]));

    // Temporary arrays
    auto h_counts = pinned_.GetSpan<PartitionCountsT>(nidx.size(), PartitionCountsT{});
    dh::TemporaryArray<PartitionCountsT> d_counts(nidx.size(), PartitionCountsT{});

    // Partition the rows according to the operator
    SortPositionBatchUnstable( dh::ToSpan(d_batch_info), dh::ToSpan(ridx_), dh::ToSpan(ridx_tmp_),
                       dh::ToSpan(d_counts), total_rows,op, 
                      streams_[1]);
    dh::safe_cuda(
        cudaMemcpyAsync(h_counts.data(), d_counts.data().get(),
                        sizeof(decltype(d_counts)::value_type) * d_counts.size(),
                        cudaMemcpyDefault, streams_[1]));

    dh::safe_cuda(cudaStreamSynchronize(streams_[1]));

    // Update segments
    for (int i = 0; i < nidx.size(); i++) {
      auto segment = ridx_segments_.at(nidx[i]);
      auto left_count = h_counts[i].first;
      CHECK_LE(left_count, segment.Size());
      CHECK_GE(left_count, 0);
      ridx_segments_.resize(std::max(static_cast<bst_node_t>(ridx_segments_.size()),
                                     std::max(left_nidx[i], right_nidx[i]) + 1));
      ridx_segments_[left_nidx[i]] = Segment(segment.begin, segment.begin + left_count);
      ridx_segments_[right_nidx[i]] = Segment(segment.begin + left_count, segment.end);
    }
  }
};
};  // namespace tree
};  // namespace xgboost
