/*!
 * Copyright 2017-2022 XGBoost contributors
 */
#pragma once
#include <thrust/execution_policy.h>

#include <limits>
#include <vector>

#include "../../common/device_helpers.cuh"
#include "xgboost/base.h"
#include "xgboost/context.h"
#include "xgboost/task.h"
#include "xgboost/tree_model.h"

namespace xgboost {
namespace tree {

/** \brief Used to demarcate a contiguous set of row indices associated with
 * some tree node. */
struct Segment {
  bst_uint begin{0};
  bst_uint end{0};

  Segment() = default;

  Segment(bst_uint begin, bst_uint end) : begin(begin), end(end) { CHECK_GE(end, begin); }
  __host__ __device__ size_t Size() const { return end - begin; }
};

// TODO(Rory): Can be larger. To be tuned alongside other batch operations.
static const int kMaxUpdatePositionBatchSize = 32;
template <typename OpDataT>
struct PerNodeData {
  Segment segment;
  OpDataT data;
};

template <typename BatchIterT>
__device__ __forceinline__ void AssignBatch(BatchIterT batch_info, std::size_t global_thread_idx,
                                            int* batch_idx, std::size_t* item_idx) {
  bst_uint sum = 0;
  for (int i = 0; i < kMaxUpdatePositionBatchSize; i++) {
    if (sum + batch_info[i].segment.Size() > global_thread_idx) {
      *batch_idx = i;
      *item_idx = (global_thread_idx - sum) + batch_info[i].segment.begin;
      break;
    }
    sum += batch_info[i].segment.Size();
  }
}

template <int kBlockSize, typename RowIndexT, typename OpDataT>
__global__ __launch_bounds__(kBlockSize) void SortPositionCopyKernel(
    dh::LDGIterator<PerNodeData<OpDataT>> batch_info, common::Span<RowIndexT> d_ridx,
    const common::Span<const RowIndexT> ridx_tmp, std::size_t total_rows) {
  for (auto idx : dh::GridStrideRange<std::size_t>(0, total_rows)) {
    int batch_idx;
    std::size_t item_idx;
    AssignBatch(batch_info, idx, &batch_idx, &item_idx);
    d_ridx[item_idx] = ridx_tmp[item_idx];
  }
}

// We can scan over this tuple, where the scan gives us information on how to partition inputs
// according to the flag
struct IndexFlagTuple {
  bst_uint idx;        // The location of the item we are working on in ridx_
  bst_uint flag_scan;  // This gets populated after scanning
  int batch_idx;       // Which node in the batch does this item belong to
  bool flag;           // Result of op (is this item going left?)
};

struct IndexFlagOp {
  __device__ IndexFlagTuple operator()(const IndexFlagTuple& a, const IndexFlagTuple& b) const {
    // Segmented scan - resets if we cross batch boundaries
    if (a.batch_idx == b.batch_idx) {
      // Accumulate the flags, everything else stays the same
      return {b.idx, a.flag_scan + b.flag_scan, b.batch_idx, b.flag};
    } else {
      return b;
    }
  }
};

template <typename OpDataT>
struct WriteResultsFunctor {
  dh::LDGIterator<PerNodeData<OpDataT>> batch_info;
  const bst_uint* ridx_in;
  bst_uint* ridx_out;
  bst_uint* counts;

  __device__ IndexFlagTuple operator()(const IndexFlagTuple& x) {
    std::size_t scatter_address;
    const Segment& segment = batch_info[x.batch_idx].segment;
    if (x.flag) {
      bst_uint num_previous_flagged = x.flag_scan - 1;  // -1 because inclusive scan
      scatter_address = segment.begin + num_previous_flagged;
    } else {
      bst_uint num_previous_unflagged = (x.idx - segment.begin) - x.flag_scan;
      scatter_address = segment.end - num_previous_unflagged - 1;
    }
    ridx_out[scatter_address] = ridx_in[x.idx];

    if (x.idx == (segment.end - 1)) {
      // Write out counts
      counts[x.batch_idx] = x.flag_scan;
    }

    // Discard
    return {};
  }
};

template <typename RowIndexT, typename OpT, typename OpDataT>
void SortPositionBatch(common::Span<const PerNodeData<OpDataT>> d_batch_info,
                       common::Span<RowIndexT> ridx, common::Span<RowIndexT> ridx_tmp,
                       common::Span<bst_uint> d_counts, std::size_t total_rows, OpT op,
                       dh::device_vector<int8_t>* tmp, cudaStream_t stream) {
  dh::LDGIterator<PerNodeData<OpDataT>> batch_info_itr(d_batch_info.data());
  WriteResultsFunctor<OpDataT> write_results{batch_info_itr, ridx.data(), ridx_tmp.data(),
                                             d_counts.data()};

  auto discard_write_iterator =
      thrust::make_transform_output_iterator(dh::TypedDiscard<IndexFlagTuple>(), write_results);
  auto counting = thrust::make_counting_iterator(0llu);
  auto input_iterator =
      dh::MakeTransformIterator<IndexFlagTuple>(counting, [=] __device__(size_t idx) {
        int batch_idx;
        std::size_t item_idx;
        AssignBatch(batch_info_itr, idx, &batch_idx, &item_idx);
        auto op_res = op(ridx[item_idx], batch_info_itr[batch_idx].data);
        return IndexFlagTuple{static_cast<bst_uint>(item_idx), op_res, batch_idx, op_res};
      });
  size_t temp_bytes = 0;
  if (tmp->empty()) {
    cub::DeviceScan::InclusiveScan(nullptr, temp_bytes, input_iterator, discard_write_iterator,
                                   IndexFlagOp(), total_rows, stream);
    tmp->resize(temp_bytes);
  }
  temp_bytes = tmp->size();
  cub::DeviceScan::InclusiveScan(tmp->data().get(), temp_bytes, input_iterator,
                                 discard_write_iterator, IndexFlagOp(), total_rows, stream);

  constexpr int kBlockSize = 256;

  // Value found by experimentation
  const int kItemsThread = 12;
  const int grid_size = xgboost::common::DivRoundUp(total_rows, kBlockSize * kItemsThread);

  SortPositionCopyKernel<kBlockSize, RowIndexT, OpDataT>
      <<<grid_size, kBlockSize, 0, stream>>>(batch_info_itr, ridx, ridx_tmp, total_rows);
}

struct NodePositionInfo {
  Segment segment;
  bst_node_t left_child = -1;
  bst_node_t right_child = -1;
  __device__ bool IsLeaf() { return left_child == -1; }
};

__device__ __forceinline__ int GetPositionFromSegments(std::size_t idx,
                                                       const NodePositionInfo* d_node_info) {
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
    const common::Span<const RowIndexT> d_ridx, common::Span<bst_node_t> d_out_position, OpT op) {
  for (auto idx : dh::GridStrideRange<std::size_t>(0, d_ridx.size())) {
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
  static constexpr bst_node_t kIgnoredTreePosition = -1;

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
  dh::device_vector<int8_t> tmp_;
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

  /**
   * \brief Updates the tree position for set of training instances being split
   * into left and right child nodes. Accepts a user-defined lambda specifying
   * which branch each training instance should go down.
   *
   * \tparam  UpdatePositionOpT
   * \tparam  OpDataT
   * \param nidx        The index of the nodes being split.
   * \param left_nidx   The left child indices.
   * \param right_nidx  The right child indices.
   * \param op_data     User-defined data provided as the second argument to op
   * \param op          Device lambda with the row index as the first argument and op_data as the
   * second. Returns true if this training instance goes on the left partition.
   */
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
    for (size_t i = 0; i < nidx.size(); i++) {
      h_batch_info[i] = {ridx_segments_.at(nidx.at(i)).segment, op_data.at(i)};
      total_rows += ridx_segments_.at(nidx.at(i)).segment.Size();
    }
    dh::safe_cuda(cudaMemcpyAsync(d_batch_info.data().get(), h_batch_info.data(),
                                  h_batch_info.size() * sizeof(PerNodeData<OpDataT>),
                                  cudaMemcpyDefault, stream_));

    // Temporary arrays
    auto h_counts = pinned_.GetSpan<bst_uint>(nidx.size(), 0);
    dh::TemporaryArray<bst_uint> d_counts(nidx.size(), 0);

    // Partition the rows according to the operator
    SortPositionBatch<RowIndexT, UpdatePositionOpT, OpDataT>(
        dh::ToSpan(d_batch_info), dh::ToSpan(ridx_), dh::ToSpan(ridx_tmp_), dh::ToSpan(d_counts),
        total_rows, op, &tmp_, stream_);
    dh::safe_cuda(cudaMemcpyAsync(h_counts.data(), d_counts.data().get(), h_counts.size_bytes(),
                                  cudaMemcpyDefault, stream_));
    // TODO(Rory): this synchronisation hurts performance a lot
    // Future optimisation should find a way to skip this
    dh::safe_cuda(cudaStreamSynchronize(stream_));

    // Update segments
    for (size_t i = 0; i < nidx.size(); i++) {
      auto segment = ridx_segments_.at(nidx[i]).segment;
      auto left_count = h_counts[i];
      CHECK_LE(left_count, segment.Size());
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
  void FinalisePosition(common::Span<bst_node_t> d_out_position, FinalisePositionOpT op) {
    dh::TemporaryArray<NodePositionInfo> d_node_info_storage(ridx_segments_.size());
    dh::safe_cuda(cudaMemcpyAsync(d_node_info_storage.data().get(), ridx_segments_.data(),
                                  sizeof(NodePositionInfo) * ridx_segments_.size(),
                                  cudaMemcpyDefault, stream_));

    constexpr int kBlockSize = 512;
    const int kItemsThread = 8;
    const int grid_size = xgboost::common::DivRoundUp(ridx_.size(), kBlockSize * kItemsThread);
    common::Span<const RowIndexT> d_ridx(ridx_.data().get(), ridx_.size());
    FinalisePositionKernel<kBlockSize><<<grid_size, kBlockSize, 0, stream_>>>(
        dh::ToSpan(d_node_info_storage), d_ridx, d_out_position, op);
  }
};

};  // namespace tree
};  // namespace xgboost
