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

// TODO(Rory): Can be larger. To be tuned alongside other batch operations.
static const int kMaxUpdatePositionBatchSize = 32;
template <typename OpDataT>
struct PerNodeData {
  Segment segment;
  OpDataT data;
};

template <typename OpDataT>
__device__ __forceinline__ void AssignBatch(const PerNodeData<OpDataT> *batch_info,
                            std::size_t global_thread_idx, int* batch_idx, std::size_t* item_idx) {
  std::size_t sum = 0;
  for (int16_t i = 0; i < kMaxUpdatePositionBatchSize; i++) {
    if (sum + batch_info[i].segment.Size() > global_thread_idx) {
      *batch_idx = i;
      *item_idx = (global_thread_idx - sum) + batch_info[i].segment.begin;
      break;
    }
    sum += batch_info[i].segment.Size();
  }
}

__forceinline__ __device__ uint32_t __lanemask_lt() { return ((uint32_t)1 << cub::LaneId()) - 1; }

__forceinline__ __device__ uint32_t AtomicIncrement(PartitionCountsT* d_counts, bool go_left,
                                                int16_t batch_idx) {
  int mask = __activemask();
  int leader = __ffs(mask) - 1;
  uint32_t prefix = __popc(mask & __lanemask_lt());
  bool group_is_contiguous = __all_sync(mask, batch_idx == __shfl_sync(mask, batch_idx, leader));
  // If all threads here are working on the same node
  // we can do a more efficient reduction with warp intrinsics
  if (group_is_contiguous) {
    uint32_t ballot = __ballot_sync(mask, go_left);
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

    return go_left ? global_left_count + local_left_count : global_right_count + local_right_count;

  } else {
    auto address = go_left ? &d_counts->first : &d_counts->second;
    return atomicAdd(address, 1);
  }
}

template <typename OpDataT>
struct SharedStorage {
  PerNodeData<OpDataT> data[kMaxUpdatePositionBatchSize];
  // Collectively load from global memory into shared memory
  template <int kBlockSize>
  __device__ const PerNodeData<OpDataT>* BlockLoad(
      const common::Span<const PerNodeData<OpDataT>> d_batch_info) {
    for (int i = threadIdx.x; i < d_batch_info.size(); i += kBlockSize) {
      data[i] = d_batch_info.data()[i];
    }
    __syncthreads();
    return data;
  }
};

template <int kBlockSize, typename RowIndexT, typename OpT,
          typename OpDataT>
__global__ __launch_bounds__(kBlockSize) void SortPositionBatchUnstableKernel(
    const common::Span<const PerNodeData<OpDataT>> d_batch_info, common::Span<RowIndexT> d_ridx,
    common::Span<RowIndexT> ridx_tmp, common::Span<PartitionCountsT> counts, OpT op,
    std::size_t total_rows) {
  // Initialise shared memory this way to avoid calling constructors
  __shared__ cub::Uninitialized<SharedStorage<OpDataT>> shared;
  auto batch_info = shared.Alias().BlockLoad<kBlockSize>(d_batch_info);

  for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_rows;
       idx += blockDim.x * gridDim.x) {
    int batch_idx;
    std::size_t item_idx;
    AssignBatch(batch_info, idx, &batch_idx, &item_idx);
    auto ridx = d_ridx[item_idx];
    auto op_res = op(ridx, batch_info[batch_idx].data);
    auto current_num_items = AtomicIncrement(&counts.data()[batch_idx], op_res, batch_idx);
    auto segment = batch_info[batch_idx].segment;
    auto destination_address =
        op_res ? segment.begin + current_num_items : segment.end - current_num_items - 1;
    ridx_tmp[destination_address] = ridx;
  }
}

template <int kBlockSize, typename RowIndexT, typename OpDataT>
__global__ __launch_bounds__(kBlockSize) void SortPositionCopyKernel(
    const common::Span<const PerNodeData<OpDataT>> d_batch_info, common::Span<RowIndexT> d_ridx,
    common::Span<RowIndexT> ridx_tmp,
    std::size_t total_rows) {
      
  __shared__ cub::Uninitialized<SharedStorage<OpDataT>> shared;
  auto batch_info = shared.Alias().BlockLoad<kBlockSize>(d_batch_info);
  
  for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_rows;
       idx += blockDim.x * gridDim.x) {
    int batch_idx;
    std::size_t item_idx;
    AssignBatch(batch_info, idx,&batch_idx, &item_idx);
    d_ridx[item_idx] = ridx_tmp[item_idx];
  }
}

template <typename RowIndexT, typename OpT, typename OpDataT>
void SortPositionBatchUnstable(const common::Span<const PerNodeData<OpDataT>> batch_info,
                               common::Span<RowIndexT> ridx, common::Span<RowIndexT> ridx_tmp,
                               common::Span<PartitionCountsT> d_counts, std::size_t total_rows,
                               OpT op, cudaStream_t stream) {
  CHECK_LE(batch_info.size(), kMaxUpdatePositionBatchSize);
  constexpr int kBlockSize = 256;

  // Value found by experimentation
  const int kItemsThread = 12;
  const int grid_size = xgboost::common::DivRoundUp(total_rows, kBlockSize * kItemsThread);

  SortPositionBatchUnstableKernel<kBlockSize>
      <<<grid_size, kBlockSize, 0, stream>>>(batch_info, ridx, ridx_tmp,d_counts, op, total_rows);

  SortPositionCopyKernel<kBlockSize>
      <<<grid_size, kBlockSize, 0, stream>>>(batch_info, ridx, ridx_tmp, total_rows);
}

struct NodePositionInfo {
  Segment segment;
  int left_child = -1;
  int right_child = -1;
  __device__ bool IsLeaf() { return left_child == -1; }
};

__device__ __forceinline__ int GetPositionFromSegments(std::size_t idx, const NodePositionInfo* d_node_info) {
  int position = 0;
  NodePositionInfo node = d_node_info[position];
  while (!node.IsLeaf()) {
    NodePositionInfo left = d_node_info[node.left_child];
    NodePositionInfo right = d_node_info[node.right_child];
    if (idx >= left.segment.begin && idx < left.segment.end) {
      position = node.left_child;
      node = left;
    } else if (idx >= right.segment.begin && idx < right.segment.end) {
      position = node.right_child;
      node = right;
    } else {
      KERNEL_CHECK(false);
    }
  }
  return position;
}

template <int kBlockSize, typename RowIndexT, typename OpT>
__global__ __launch_bounds__(kBlockSize) void FinalisePositionKernel(
    const common::Span<const NodePositionInfo> d_node_info,
    const common::Span<const RowIndexT> d_ridx,common::Span<bst_node_t> d_out_position, OpT op) {
  for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < d_ridx.size();
       idx += blockDim.x * gridDim.x) {
    auto position = GetPositionFromSegments(idx, d_node_info.data());
    RowIndexT ridx = d_ridx[idx];
    bst_node_t new_position = op(ridx, position);
    d_out_position[ridx] = new_position;
  }
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

  std::vector<NodePositionInfo> ridx_segments_;
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
  cudaStream_t stream_;

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

    auto h_batch_info = pinned2_.GetSpan<PerNodeData<OpDataT>>(nidx.size());
    dh::TemporaryArray<PerNodeData<OpDataT>> d_batch_info(nidx.size());

    std::size_t total_rows = 0;
    for (int i = 0; i < nidx.size(); i++) {
      h_batch_info[i] = {ridx_segments_.at(nidx.at(i)).segment,
                         op_data.at(i)};
      total_rows += ridx_segments_.at(nidx.at(i)).segment.Size();
    }
    dh::safe_cuda(cudaMemcpyAsync(d_batch_info.data().get(), h_batch_info.data(),
                                  h_batch_info.size() * sizeof(PerNodeData<OpDataT>),
                                  cudaMemcpyDefault, stream_));

    // Temporary arrays
    auto h_counts = pinned_.GetSpan<PartitionCountsT>(nidx.size(), PartitionCountsT{});
    dh::TemporaryArray<PartitionCountsT> d_counts(nidx.size(), PartitionCountsT{});

    // Partition the rows according to the operator
    SortPositionBatchUnstable(common::Span<const PerNodeData<OpDataT>>(
                                  d_batch_info.data().get(), d_batch_info.size()),
                              dh::ToSpan(ridx_), dh::ToSpan(ridx_tmp_),dh::ToSpan(d_counts),
                              total_rows, op, stream_);
    dh::safe_cuda(
        cudaMemcpyAsync(h_counts.data(), d_counts.data().get(),
                        sizeof(decltype(d_counts)::value_type) * d_counts.size(),
                        cudaMemcpyDefault, stream_));

    dh::safe_cuda(cudaStreamSynchronize(stream_));

    // Update segments
    for (int i = 0; i < nidx.size(); i++) {
      auto segment = ridx_segments_.at(nidx[i]).segment;
      auto left_count = h_counts[i].first;
      CHECK_LE(left_count, segment.Size());
      CHECK_GE(left_count, 0);
      ridx_segments_.resize(std::max(static_cast<bst_node_t>(ridx_segments_.size()),
                                     std::max(left_nidx[i], right_nidx[i]) + 1));
      ridx_segments_[nidx[i]] = NodePositionInfo{segment, left_nidx[i], right_nidx[i]};
      ridx_segments_[left_nidx[i]] =
          NodePositionInfo{Segment(segment.begin, segment.begin + left_count)};
      ridx_segments_[right_nidx[i]] =
          NodePositionInfo{Segment(segment.begin + left_count, segment.end)};
    }
  }

   /**
   * \brief Finalise the position of all training instances after tree construction is
   * complete. Does not update any other meta information in this data structure, so
   * should only be used at the end of training.
   *
   *   When the task requires update leaf, this function will copy the node index into
   *   p_out_position. The index is negated if it's being sampled in current iteration.
   *
   * \param p_out_position Node index for each row.
   * \param op Device lambda. Should provide the row index and current position as an
   *           argument and return the new position for this training instance.
   * \param sampled A device lambda to inform the partitioner whether a row is sampled.
   */
  template <typename FinalisePositionOpT>
  void FinalisePosition(
                        common::Span<bst_node_t> d_out_position, FinalisePositionOpT op) {
    dh::TemporaryArray<NodePositionInfo> d_node_info_storage(ridx_segments_.size());
    dh::safe_cuda(cudaMemcpyAsync(d_node_info_storage.data().get(), ridx_segments_.data(),
                                  sizeof(NodePositionInfo) * ridx_segments_.size(),
                                  cudaMemcpyDefault, stream_));

    auto d_node_info = d_node_info_storage.data().get();

    auto current_position = [=] __device__(std::size_t idx) {
      int position = 0;
      NodePositionInfo node = d_node_info[position];
      while (!node.IsLeaf()) {
        NodePositionInfo left = d_node_info[node.left_child];
        NodePositionInfo right = d_node_info[node.right_child];
        if (idx >= left.segment.begin && idx < left.segment.end) {
          position = node.left_child;
          node = left;
        } else if (idx >= right.segment.begin && idx < right.segment.end) {
          position = node.right_child;
          node = right;
        } else {
          KERNEL_CHECK(false);
        }
      }
      return position;
    };

  constexpr int kBlockSize = 256;

  // Value found by experimentation
  const int kItemsThread = 8;
  const int grid_size = xgboost::common::DivRoundUp(ridx_.size(), kBlockSize * kItemsThread);
  common::Span<const RowIndexT> d_ridx(ridx_.data().get(), ridx_.size());
  FinalisePositionKernel<kBlockSize><<<grid_size, kBlockSize, 0, stream_>>>(
      dh::ToSpan(d_node_info_storage), d_ridx, d_out_position, op);
  }
};

};  // namespace tree
};  // namespace xgboost
