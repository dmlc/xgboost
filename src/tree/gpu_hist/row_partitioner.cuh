/**
 * Copyright 2017-2024, XGBoost contributors
 */
#pragma once
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>          // for make_counting_iterator
#include <thrust/iterator/transform_output_iterator.h>  // for make_transform_output_iterator

#include <algorithm>  // for max
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t, uint32_t
#include <vector>     // for vector

#include "../../common/cuda_context.cuh"    // for CUDAContext
#include "../../common/device_helpers.cuh"  // for MakeTransformIterator
#include "xgboost/base.h"                   // for bst_idx_t
#include "xgboost/context.h"                // for Context
#include "xgboost/span.h"                   // for Span

namespace xgboost::tree {
namespace cuda_impl {
using RowIndexT = std::uint32_t;
// TODO(Rory): Can be larger. To be tuned alongside other batch operations.
static const std::int32_t kMaxUpdatePositionBatchSize = 32;
}  // namespace cuda_impl

/**
 * @brief Used to demarcate a contiguous set of row indices associated with some tree
 *        node.
 */
struct Segment {
  cuda_impl::RowIndexT begin{0};
  cuda_impl::RowIndexT end{0};

  Segment() = default;

  Segment(cuda_impl::RowIndexT begin, cuda_impl::RowIndexT end) : begin(begin), end(end) {
    CHECK_GE(end, begin);
  }
  __host__ __device__ bst_idx_t Size() const { return end - begin; }
};

template <typename OpDataT>
struct PerNodeData {
  Segment segment;
  OpDataT data;
};

template <typename BatchIterT>
XGBOOST_DEV_INLINE void AssignBatch(BatchIterT batch_info, std::size_t global_thread_idx,
                                    int* batch_idx, std::size_t* item_idx) {
  cuda_impl::RowIndexT sum = 0;
  for (int i = 0; i < cuda_impl::kMaxUpdatePositionBatchSize; i++) {
    if (sum + batch_info[i].segment.Size() > global_thread_idx) {
      *batch_idx = i;
      *item_idx = (global_thread_idx - sum) + batch_info[i].segment.begin;
      break;
    }
    sum += batch_info[i].segment.Size();
  }
}

template <int kBlockSize, typename OpDataT>
__global__ __launch_bounds__(kBlockSize) void SortPositionCopyKernel(
    dh::LDGIterator<PerNodeData<OpDataT>> batch_info, common::Span<cuda_impl::RowIndexT> d_ridx,
    const common::Span<const cuda_impl::RowIndexT> ridx_tmp, bst_idx_t total_rows) {
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
  cuda_impl::RowIndexT idx;        // The location of the item we are working on in ridx_
  cuda_impl::RowIndexT flag_scan;  // This gets populated after scanning
  std::int32_t batch_idx;          // Which node in the batch does this item belong to
  bool flag;                       // Result of op (is this item going left?)
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

// Scatter from `ridx_in` to `ridx_out`.
template <typename OpDataT>
struct WriteResultsFunctor {
  dh::LDGIterator<PerNodeData<OpDataT>> batch_info;
  cuda_impl::RowIndexT const* ridx_in;
  cuda_impl::RowIndexT* ridx_out;
  cuda_impl::RowIndexT* counts;

  __device__ IndexFlagTuple operator()(IndexFlagTuple const& x) {
    cuda_impl::RowIndexT scatter_address;
    // Get the segment that this row belongs to.
    const Segment& segment = batch_info[x.batch_idx].segment;
    if (x.flag) {
      // Go left.
      cuda_impl::RowIndexT num_previous_flagged = x.flag_scan - 1;  // -1 because inclusive scan
      scatter_address = segment.begin + num_previous_flagged;
    } else {
      cuda_impl::RowIndexT num_previous_unflagged = (x.idx - segment.begin) - x.flag_scan;
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

/**
 * @param d_batch_info Node data, with the size of the input number of nodes.
 */
template <typename OpT, typename OpDataT>
void SortPositionBatch(Context const* ctx, common::Span<const PerNodeData<OpDataT>> d_batch_info,
                       common::Span<cuda_impl::RowIndexT> ridx,
                       common::Span<cuda_impl::RowIndexT> ridx_tmp,
                       common::Span<cuda_impl::RowIndexT> d_counts, bst_idx_t total_rows, OpT op,
                       dh::DeviceUVector<int8_t>* tmp) {
  dh::LDGIterator<PerNodeData<OpDataT>> batch_info_itr(d_batch_info.data());
  WriteResultsFunctor<OpDataT> write_results{batch_info_itr, ridx.data(), ridx_tmp.data(),
                                             d_counts.data()};

  auto discard_write_iterator =
      thrust::make_transform_output_iterator(dh::TypedDiscard<IndexFlagTuple>(), write_results);
  auto counting = thrust::make_counting_iterator(0llu);
  auto input_iterator =
      dh::MakeTransformIterator<IndexFlagTuple>(counting, [=] __device__(std::size_t idx) {
        int nidx_in_batch;
        std::size_t item_idx;
        AssignBatch(batch_info_itr, idx, &nidx_in_batch, &item_idx);
        auto go_left = op(ridx[item_idx], nidx_in_batch, batch_info_itr[nidx_in_batch].data);
        return IndexFlagTuple{static_cast<cuda_impl::RowIndexT>(item_idx), go_left, nidx_in_batch,
                              go_left};
      });
  // Avoid using int as the offset type
  std::size_t n_bytes = 0;
  if (tmp->empty()) {
    auto ret =
        cub::DispatchScan<decltype(input_iterator), decltype(discard_write_iterator), IndexFlagOp,
                          cub::NullType, std::int64_t>::Dispatch(nullptr, n_bytes, input_iterator,
                                                                 discard_write_iterator,
                                                                 IndexFlagOp{}, cub::NullType{},
                                                                 total_rows,
                                                                 ctx->CUDACtx()->Stream());
    dh::safe_cuda(ret);
    tmp->resize(n_bytes);
  }
  n_bytes = tmp->size();
  auto ret =
      cub::DispatchScan<decltype(input_iterator), decltype(discard_write_iterator), IndexFlagOp,
                        cub::NullType, std::int64_t>::Dispatch(tmp->data(), n_bytes, input_iterator,
                                                               discard_write_iterator,
                                                               IndexFlagOp{}, cub::NullType{},
                                                               total_rows,
                                                               ctx->CUDACtx()->Stream());
  dh::safe_cuda(ret);

  constexpr int kBlockSize = 256;

  // Value found by experimentation
  const int kItemsThread = 12;
  std::uint32_t const kGridSize =
      xgboost::common::DivRoundUp(total_rows, kBlockSize * kItemsThread);
  dh::LaunchKernel{kGridSize, kBlockSize, 0, ctx->CUDACtx()->Stream()}(
      SortPositionCopyKernel<kBlockSize, OpDataT>, batch_info_itr, ridx, ridx_tmp, total_rows);
}

struct NodePositionInfo {
  Segment segment;
  bst_node_t left_child = -1;
  bst_node_t right_child = -1;
  __device__ bool IsLeaf() { return left_child == -1; }
};

XGBOOST_DEV_INLINE int GetPositionFromSegments(std::size_t idx,
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

template <int kBlockSize, typename OpT>
__global__ __launch_bounds__(kBlockSize) void FinalisePositionKernel(
    common::Span<const NodePositionInfo> d_node_info, bst_idx_t base_ridx,
    common::Span<const cuda_impl::RowIndexT> d_ridx, common::Span<bst_node_t> d_out_position,
    OpT op) {
  for (auto idx : dh::GridStrideRange<std::size_t>(0, d_ridx.size())) {
    auto position = GetPositionFromSegments(idx, d_node_info.data());
    cuda_impl::RowIndexT ridx = d_ridx[idx] - base_ridx;
    bst_node_t new_position = op(ridx, position);
    d_out_position[ridx] = new_position;
  }
}

/** \brief Class responsible for tracking subsets of rows as we add splits and
 * partition training rows into different leaf nodes. */
class RowPartitioner {
 public:
  using RowIndexT = cuda_impl::RowIndexT;

 private:
  /**
   * In here if you want to find the rows belong to a node nid, first you need to get the
   * indices segment from ridx_segments[nid], then get the row index that represents
   * position of row in input data X.  `RowPartitioner::GetRows` would be a good starting
   * place to get a sense what are these vector storing.
   *
   * node id -> segment -> indices of rows belonging to node
   */

  /** @brief Range of row index for each node, pointers into ridx below. */
  std::vector<NodePositionInfo> ridx_segments_;
  /**
   * @brief mapping for node id -> rows.
   *
   * This looks like:
   * node id  |    1    |    2   |
   * rows idx | 3, 5, 1 | 13, 31 |
   */
  dh::DeviceUVector<RowIndexT> ridx_;
  // Staging area for sorting ridx
  dh::DeviceUVector<RowIndexT> ridx_tmp_;
  dh::DeviceUVector<int8_t> tmp_;
  dh::PinnedMemory pinned_;
  dh::PinnedMemory pinned2_;
  bst_node_t n_nodes_{0};  // Counter for internal checks.

 public:
  /**
   * @param ctx Context for device ordinal and stream.
   * @param n_samples The number of samples in each batch.
   * @param base_rowid The base row index for the current batch.
   */
  RowPartitioner() = default;
  void Reset(Context const* ctx, bst_idx_t n_samples, bst_idx_t base_rowid);

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
  common::Span<const RowIndexT> GetRows() const;
  /**
   * @brief Get the number of rows in this partitioner.
   */
  std::size_t Size() const { return this->GetRows().size(); }

  [[nodiscard]] bst_node_t GetNumNodes() const { return n_nodes_; }

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
  void UpdatePositionBatch(Context const* ctx, const std::vector<bst_node_t>& nidx,
                           const std::vector<bst_node_t>& left_nidx,
                           const std::vector<bst_node_t>& right_nidx,
                           const std::vector<OpDataT>& op_data, UpdatePositionOpT op) {
    if (nidx.empty()) {
      return;
    }

    CHECK_EQ(nidx.size(), left_nidx.size());
    CHECK_EQ(nidx.size(), right_nidx.size());
    CHECK_EQ(nidx.size(), op_data.size());
    this->n_nodes_ += (left_nidx.size() + right_nidx.size());

    auto h_batch_info = pinned2_.GetSpan<PerNodeData<OpDataT>>(nidx.size());
    dh::TemporaryArray<PerNodeData<OpDataT>> d_batch_info(nidx.size());

    std::size_t total_rows = 0;
    for (size_t i = 0; i < nidx.size(); i++) {
      h_batch_info[i] = {ridx_segments_.at(nidx.at(i)).segment, op_data.at(i)};
      total_rows += ridx_segments_.at(nidx.at(i)).segment.Size();
    }
    dh::safe_cuda(cudaMemcpyAsync(d_batch_info.data().get(), h_batch_info.data(),
                                  h_batch_info.size() * sizeof(PerNodeData<OpDataT>),
                                  cudaMemcpyDefault, ctx->CUDACtx()->Stream()));

    // Temporary arrays
    auto h_counts = pinned_.GetSpan<RowIndexT>(nidx.size());
    // Must initialize with 0 as 0 count is not written in the kernel.
    dh::TemporaryArray<RowIndexT> d_counts(nidx.size(), 0);

    // Partition the rows according to the operator
    SortPositionBatch<UpdatePositionOpT, OpDataT>(ctx, dh::ToSpan(d_batch_info), dh::ToSpan(ridx_),
                                                  dh::ToSpan(ridx_tmp_), dh::ToSpan(d_counts),
                                                  total_rows, op, &tmp_);
    dh::safe_cuda(cudaMemcpyAsync(h_counts.data(), d_counts.data().get(), h_counts.size_bytes(),
                                  cudaMemcpyDefault, ctx->CUDACtx()->Stream()));
    // TODO(Rory): this synchronisation hurts performance a lot
    // Future optimisation should find a way to skip this
    ctx->CUDACtx()->Stream().Sync();

    // Update segments
    for (std::size_t i = 0; i < nidx.size(); i++) {
      auto segment = ridx_segments_.at(nidx[i]).segment;
      auto left_count = h_counts[i];
      CHECK_LE(left_count, segment.Size());
      ridx_segments_.resize(std::max(static_cast<bst_node_t>(ridx_segments_.size()),
                                     std::max(left_nidx[i], right_nidx[i]) + 1));
      ridx_segments_[nidx[i]] = NodePositionInfo{segment, left_nidx[i], right_nidx[i]};
      ridx_segments_[left_nidx[i]] =
          NodePositionInfo{Segment{segment.begin, segment.begin + left_count}};
      ridx_segments_[right_nidx[i]] =
          NodePositionInfo{Segment{segment.begin + left_count, segment.end}};
    }
  }

  /**
   * @brief Finalise the position of all training instances after tree construction is
   * complete. Does not update any other meta information in this data structure, so
   * should only be used at the end of training.
   *
   * @param p_out_position Node index for each row.
   * @param op Device lambda. Should provide the row index and current position as an
   *           argument and return the new position for this training instance.
   */
  template <typename FinalisePositionOpT>
  void FinalisePosition(Context const* ctx, common::Span<bst_node_t> d_out_position,
                        bst_idx_t base_ridx, FinalisePositionOpT op) const {
    dh::TemporaryArray<NodePositionInfo> d_node_info_storage(ridx_segments_.size());
    dh::safe_cuda(cudaMemcpyAsync(d_node_info_storage.data().get(), ridx_segments_.data(),
                                  sizeof(NodePositionInfo) * ridx_segments_.size(),
                                  cudaMemcpyDefault, ctx->CUDACtx()->Stream()));

    constexpr std::uint32_t kBlockSize = 512;
    const int kItemsThread = 8;
    const std::uint32_t grid_size =
        xgboost::common::DivRoundUp(ridx_.size(), kBlockSize * kItemsThread);
    common::Span<RowIndexT const> d_ridx{ridx_.data(), ridx_.size()};
    dh::LaunchKernel{grid_size, kBlockSize, 0, ctx->CUDACtx()->Stream()}(
        FinalisePositionKernel<kBlockSize, FinalisePositionOpT>, dh::ToSpan(d_node_info_storage),
        base_ridx, d_ridx, d_out_position, op);
  }
};
};  // namespace xgboost::tree
