/**
 * Copyright 2017-2025, XGBoost contributors
 */
#pragma once
#include <thrust/iterator/counting_iterator.h>          // for make_counting_iterator
#include <thrust/iterator/transform_output_iterator.h>  // for make_transform_output_iterator

#include <algorithm>        // for max
#include <cstddef>          // for size_t
#include <cstdint>          // for int32_t, uint32_t
#include <cuda/functional>  // for proclaim_return_type
#include <vector>           // for vector

#include "../../common/cuda_context.cuh"    // for CUDAContext
#include "../../common/device_helpers.cuh"  // for MakeTransformIterator
#include "xgboost/base.h"                   // for bst_idx_t
#include "xgboost/context.h"                // for Context
#include "xgboost/span.h"                   // for Span

namespace xgboost::tree {
namespace cuda_impl {
using RowIndexT = std::uint32_t;
// TODO(Rory): Can be larger. To be tuned alongside other batch operations.
inline constexpr std::int32_t kMaxUpdatePositionBatchSize = 32;
}  // namespace cuda_impl

/**
 * @brief Used to demarcate a contiguous set of row indices associated with some tree
 *        node.
 */
struct Segment {
  cuda_impl::RowIndexT begin{0};
  cuda_impl::RowIndexT end{0};

  Segment() = default;

  XGBOOST_DEV_INLINE Segment(cuda_impl::RowIndexT begin, cuda_impl::RowIndexT end)
      : begin(begin), end(end) {
#if !defined(__CUDA_ARCH__)
    CHECK_GE(end, begin);
#endif  // !defined(__CUDA_ARCH__)
  }
  [[nodiscard]] XGBOOST_DEVICE bst_idx_t Size() const { return end - begin; }
};

template <typename OpDataT>
struct PerNodeData {
  Segment segment;
  OpDataT data;
};

/**
 * @param global_thread_idx In practice, the row index within the total number of rows for
 *        this node batch.
 * @param batch_idx The nidx within this node batch (not the actual node index in a tree).
 * @param item_idx The resulting global row index (without accounting for base_rowid). This maps the
 *        row index within the node batch back to the global row index.
 */
template <typename T>
XGBOOST_DEV_INLINE bool AssignBatch(dh::LDGIterator<T> const& batch_info_iter,
                                    std::size_t global_thread_idx, std::size_t batch_size,
                                    int* batch_idx, std::size_t* item_idx) {
  cuda_impl::RowIndexT sum = 0;
  // Search for the nidx in batch and the corresponding global row index, exit once found.
  for (std::size_t i = 0; i < batch_size; i++) {
    if (sum + batch_info_iter[i].segment.Size() > global_thread_idx) {
      *batch_idx = i;
      // the beginning of the segment plus the offset into that segment
      *item_idx = (global_thread_idx - sum) + batch_info_iter[i].segment.begin;
      return true;
    }
    sum += batch_info_iter[i].segment.Size();
  }
  return false;
}

/**
 * @param total_rows The total number of rows for this batch of nodes.
 */
template <int kBlockSize, typename OpDataT>
__global__ __launch_bounds__(kBlockSize) void SortPositionCopyKernel(
    dh::LDGIterator<PerNodeData<OpDataT>> batch_info_iter,
    common::Span<cuda_impl::RowIndexT> d_ridx,
    common::Span<cuda_impl::RowIndexT const> const ridx_tmp, std::size_t batch_size) {
  for (auto idx : dh::GridStrideRange<std::size_t>(0, ridx_tmp.size())) {
    std::int32_t batch_idx;  // unused
    std::size_t item_idx = std::numeric_limits<std::size_t>::max();
    if (AssignBatch(batch_info_iter, idx, batch_size, &batch_idx, &item_idx)) {
      d_ridx[item_idx] = ridx_tmp[item_idx];
    }
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
    if (x.batch_idx < 0) {
      return {};
    }
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
                       common::Span<cuda_impl::RowIndexT> d_counts, bst_idx_t /*total_rows*/, OpT op,
                       dh::DeviceUVector<int8_t>* tmp) {
  dh::LDGIterator<PerNodeData<OpDataT>> batch_info_itr(d_batch_info.data());
  WriteResultsFunctor<OpDataT> write_results{batch_info_itr, ridx.data(), ridx_tmp.data(),
                                             d_counts.data()};

  auto discard_write_iterator =
      thrust::make_transform_output_iterator(dh::TypedDiscard<IndexFlagTuple>(), write_results);
  auto counting = thrust::make_counting_iterator(0llu);
  auto input_iterator = dh::MakeTransformIterator<IndexFlagTuple>(
      counting, cuda::proclaim_return_type<IndexFlagTuple>([=] __device__(std::size_t idx) {
        std::int32_t nidx_in_batch;
        std::size_t item_idx;
        if (!AssignBatch(batch_info_itr, idx, d_batch_info.size(), &nidx_in_batch, &item_idx)) {
          return IndexFlagTuple{0, 0, -1, false};
        }
        auto go_left = op(ridx[item_idx], batch_info_itr[nidx_in_batch].data);
        return IndexFlagTuple{static_cast<cuda_impl::RowIndexT>(item_idx), go_left, nidx_in_batch,
                              go_left};
      }));
  // Reach down to the dispatch function to avoid using int as the offset type.
  std::size_t n_bytes = 0;
  if (tmp->empty()) {
    // The size of temporary storage is calculated based on the total number of
    // rows. Since the root node has all the rows, subsequence allocatioin must be smaller
    // than the root node. As a result, we can calculate this once and reuse it throughout
    // the iteration.
    auto ret =
        cub::DispatchScan<decltype(input_iterator), decltype(discard_write_iterator), IndexFlagOp,
                          cub::NullType, std::uint64_t>::Dispatch(nullptr, n_bytes, input_iterator,
                                                                  discard_write_iterator,
                                                                  IndexFlagOp{}, cub::NullType{},
                                                                  static_cast<std::uint64_t>(
                                                                      ridx.size()),
                                                                  ctx->CUDACtx()->Stream());
    dh::safe_cuda(ret);
    tmp->resize(n_bytes);
  }
  n_bytes = tmp->size();
  auto ret =
      cub::DispatchScan<decltype(input_iterator), decltype(discard_write_iterator), IndexFlagOp,
                        cub::NullType, std::uint64_t>::Dispatch(tmp->data(), n_bytes,
                                                                input_iterator,
                                                                discard_write_iterator,
                                                                IndexFlagOp{}, cub::NullType{},
                                                                static_cast<std::uint64_t>(
                                                                    ridx.size()),
                                                                ctx->CUDACtx()->Stream());
  dh::safe_cuda(ret);

  constexpr int kBlockSize = 256;

  // Value found by experimentation
  const int kItemsThread = 12;
  std::uint32_t const kGridSize =
      xgboost::common::DivRoundUp(ridx.size(), kBlockSize * kItemsThread);
  dh::LaunchKernel{kGridSize, kBlockSize, 0, ctx->CUDACtx()->Stream()}(
      SortPositionCopyKernel<kBlockSize, OpDataT>, batch_info_itr, ridx, ridx_tmp,
      d_batch_info.size());
}

struct NodePositionInfo {
  Segment segment;
  bst_node_t left_child = -1;
  bst_node_t right_child = -1;
  [[nodiscard]] XGBOOST_DEVICE bool IsLeaf() const { return left_child == -1; }
};

// A row-index range whose bounds are read on the device.  Host callers deliberately use the
// full allocation as the launch bound; device callers use the current node segment.
struct DeviceRows {
  cuda_impl::RowIndexT const* ridx;
  NodePositionInfo const* segments;
  bst_node_t nidx;
  std::size_t max_size;

  XGBOOST_DEV_INLINE cuda_impl::RowIndexT operator[](std::size_t i) const {
    return ridx[segments[nidx].segment.begin + i];
  }
  XGBOOST_DEV_INLINE std::size_t size() const {
#if defined(__CUDA_ARCH__)
    return segments[nidx].segment.Size();
#else
    return max_size;
#endif  // defined(__CUDA_ARCH__)
  }
  XGBOOST_DEV_INLINE cuda_impl::RowIndexT const* data() const {
    return ridx + segments[nidx].segment.begin;
  }
};

template <typename OpDataT>
__global__ void BuildBatchInfoKernel(common::Span<NodePositionInfo const> d_ridx_segments,
                                     common::Span<bst_node_t const> nidx,
                                     common::Span<OpDataT const> op_data,
                                     common::Span<PerNodeData<OpDataT>> d_batch_info);

__global__ void UpdateSegmentsKernel(common::Span<NodePositionInfo> d_ridx_segments,
                                     common::Span<bst_node_t const> nidx,
                                     common::Span<bst_node_t const> left_nidx,
                                     common::Span<bst_node_t const> right_nidx,
                                     common::Span<cuda_impl::RowIndexT const> counts);

struct LeafInfo {
  bst_node_t nidx;
  NodePositionInfo node;
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
    auto global_ridx = d_ridx[idx];
    auto local_ridx = global_ridx - base_ridx;
    bst_node_t new_position = op(global_ridx, position);
    d_out_position[local_ridx] = new_position;
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
  dh::DeviceUVector<NodePositionInfo> d_ridx_segments_;

  /**
   * @brief mapping for node id -> rows.
   *
   * This looks like:
   * node id  |    1    |    2   |
   * rows idx | 3, 5, 1 | 13, 31 |
   */
  dh::DeviceUVector<RowIndexT> ridx_;
  // Reused across split batches to avoid allocating a count buffer for every update.
  dh::DeviceUVector<RowIndexT> counts_;
  dh::DeviceUVector<int8_t> tmp_;
  dh::PinnedMemory pinned_;
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
  std::size_t Size() const { return ridx_.size(); }

  [[nodiscard]] common::Span<NodePositionInfo const> DeviceSegments() const {
    return dh::ToSpan(d_ridx_segments_);
  }
  [[nodiscard]] DeviceRows GetDeviceRows(bst_node_t nidx) const {
    return {ridx_.data(), d_ridx_segments_.data(), nidx, ridx_.size()};
  }

  [[nodiscard]] bst_node_t GetNumNodes() const { return n_nodes_; }

  /**
   * @brief Convenience method for testing.
   */
  std::vector<RowIndexT> GetRowsHost(bst_node_t nidx);

  [[nodiscard]] std::vector<LeafInfo> GetLeaves() const {
    std::vector<LeafInfo> leaves;
    bst_node_t nidx = 0;
    for (auto const& node : this->ridx_segments_) {
      if (node.IsLeaf()) {
        leaves.emplace_back(LeafInfo{nidx, node});
      }
      nidx += 1;
    }
    return leaves;
  }

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
  void UpdatePositionBatch(Context const* ctx, std::vector<bst_node_t> const& nidx,
                           std::vector<bst_node_t> const& left_nidx,
                           std::vector<bst_node_t> const& right_nidx,
                           std::vector<OpDataT> const& op_data, common::Span<RowIndexT> ridx_tmp,
                           UpdatePositionOpT op) {
    if (nidx.empty()) {
      return;
    }

    CHECK_EQ(nidx.size(), left_nidx.size());
    CHECK_EQ(nidx.size(), right_nidx.size());
    CHECK_EQ(nidx.size(), op_data.size());
    this->n_nodes_ += (left_nidx.size() + right_nidx.size());
    dh::TemporaryArray<PerNodeData<OpDataT>> d_batch_info(nidx.size());
    dh::TemporaryArray<bst_node_t> d_nidx(nidx.size());
    dh::TemporaryArray<bst_node_t> d_left_nidx(left_nidx.size());
    dh::TemporaryArray<bst_node_t> d_right_nidx(right_nidx.size());
    dh::TemporaryArray<OpDataT> d_op_data(op_data.size());
    auto stream = ctx->CUDACtx()->Stream();
    dh::safe_cuda(cudaMemcpyAsync(d_nidx.data().get(), nidx.data(), nidx.size() * sizeof(bst_node_t),
                                  cudaMemcpyDefault, stream));
    dh::safe_cuda(cudaMemcpyAsync(d_left_nidx.data().get(), left_nidx.data(),
                                  left_nidx.size() * sizeof(bst_node_t),
                                  cudaMemcpyDefault, stream));
    dh::safe_cuda(cudaMemcpyAsync(d_right_nidx.data().get(), right_nidx.data(),
                                  right_nidx.size() * sizeof(bst_node_t),
                                  cudaMemcpyDefault, stream));
    dh::safe_cuda(cudaMemcpyAsync(d_op_data.data().get(), op_data.data(),
                                  op_data.size() * sizeof(OpDataT),
                                  cudaMemcpyDefault, stream));
    auto max_nidx = *std::max_element(right_nidx.cbegin(), right_nidx.cend());
    max_nidx = std::max(max_nidx, *std::max_element(left_nidx.cbegin(), left_nidx.cend()));
    d_ridx_segments_.resize(static_cast<std::size_t>(max_nidx) + 1);
    auto d_segments = d_ridx_segments_.data();
    auto d_batch = d_batch_info.data().get();
    auto d_nidx_ptr = d_nidx.data().get();
    auto d_left_nidx_ptr = d_left_nidx.data().get();
    auto d_right_nidx_ptr = d_right_nidx.data().get();
    auto d_op_data_ptr = d_op_data.data().get();
    dh::LaunchN(nidx.size(), stream, [=] __device__(std::size_t i) {
      d_batch[i] = {d_segments[d_nidx_ptr[i]].segment, d_op_data_ptr[i]};
    });
    // Zero counts for empty segments; the scatter functor only writes a count for a non-empty
    // segment.  Keep this storage with the partitioner to avoid a device allocation per split.
    counts_.resize(nidx.size());
    dh::safe_cuda(cudaMemsetAsync(counts_.data(), 0, counts_.size() * sizeof(RowIndexT),
                                  ctx->CUDACtx()->Stream()));
    CHECK_EQ(ridx_tmp.size(), this->Size());

    // Process a sub-batch
    auto sub_batch_impl = [&](common::Span<PerNodeData<OpDataT>> d_batch_info,
                              common::Span<RowIndexT> d_counts) {
      // Partition the rows according to the operator
      SortPositionBatch<UpdatePositionOpT, OpDataT>(ctx, d_batch_info, dh::ToSpan(this->ridx_),
                                                    ridx_tmp, d_counts, ridx_tmp.size(), op,
                                                    &this->tmp_);
    };

    // Divide inputs into sub-batches.
    for (std::size_t batch_begin = 0, n = nidx.size(); batch_begin < n;
         batch_begin += cuda_impl::kMaxUpdatePositionBatchSize) {
      auto constexpr kMax = static_cast<decltype(n)>(cuda_impl::kMaxUpdatePositionBatchSize);
      auto batch_size = std::min(kMax, n - batch_begin);
      auto d_info_batch = dh::ToSpan(d_batch_info).subspan(batch_begin, batch_size);
      auto d_counts_batch = dh::ToSpan(counts_).subspan(batch_begin, batch_size);
      sub_batch_impl(d_info_batch, d_counts_batch);
    }
    auto d_counts = counts_.data();
    dh::LaunchN(nidx.size(), stream, [=] __device__(std::size_t i) {
      auto segment = d_segments[d_nidx_ptr[i]].segment;
      auto left_count = d_counts[i];
      d_segments[d_nidx_ptr[i]] = {segment, d_left_nidx_ptr[i], d_right_nidx_ptr[i]};
      d_segments[d_left_nidx_ptr[i]] =
          {Segment{segment.begin, segment.begin + left_count}};
      d_segments[d_right_nidx_ptr[i]] =
          {Segment{segment.begin + left_count, segment.end}};
    });
  }

  /**
   * @brief Finalise the position of all training instances after tree construction is
   * complete. Does not update any other meta information in this data structure, so
   * should only be used at the end of training.
   *
   * @param p_out_position Node index for each row in this batch.
   * @param op Device lambda. Receives the global row index and current position, and returns the
   *           new position for this training instance.
   */
  template <typename FinalisePositionOpT>
  void FinalisePosition(Context const* ctx, common::Span<bst_node_t> d_out_position,
                        bst_idx_t base_ridx, FinalisePositionOpT op) const {
    constexpr std::uint32_t kBlockSize = 512;
    const int kItemsThread = 8;
    const std::uint32_t grid_size =
        xgboost::common::DivRoundUp(ridx_.size(), kBlockSize * kItemsThread);
    common::Span<RowIndexT const> d_ridx{ridx_.data(), ridx_.size()};
    dh::LaunchKernel{grid_size, kBlockSize, 0, ctx->CUDACtx()->Stream()}(
        FinalisePositionKernel<kBlockSize, FinalisePositionOpT>, dh::ToSpan(d_ridx_segments_),
        base_ridx, d_ridx, d_out_position, op);
  }
};

// Partitioner for all batches, used for external memory training.
class RowPartitionerBatches {
 private:
  // Temporary buffer for sorting the samples.
  dh::DeviceUVector<cuda_impl::RowIndexT> ridx_tmp_;
  // Partitioners for each batch.
  std::vector<std::unique_ptr<RowPartitioner>> partitioners_;

 public:
  void Reset(Context const* ctx, std::vector<bst_idx_t> const& batch_ptr) {
    CHECK_GE(batch_ptr.size(), 2);
    std::size_t n_batches = batch_ptr.size() - 1;
    if (partitioners_.size() != n_batches) {
      partitioners_.clear();
    }

    bst_idx_t n_max_samples = 0;
    for (std::size_t k = 0; k < n_batches; ++k) {
      if (partitioners_.size() != n_batches) {
        // First run.
        partitioners_.emplace_back(std::make_unique<RowPartitioner>());
      }
      auto base_ridx = batch_ptr[k];
      auto n_samples = batch_ptr.at(k + 1) - base_ridx;
      partitioners_[k]->Reset(ctx, n_samples, base_ridx);
      CHECK_LE(n_samples, std::numeric_limits<cuda_impl::RowIndexT>::max());
      n_max_samples = std::max(n_samples, n_max_samples);
    }
    this->ridx_tmp_.resize(n_max_samples);
  }

  // Accessors
  [[nodiscard]] decltype(auto) operator[](std::size_t i) { return partitioners_[i]; }
  decltype(auto) At(std::size_t i) { return partitioners_.at(i); }
  [[nodiscard]] std::size_t Size() const { return this->partitioners_.size(); }
  decltype(auto) cbegin() const { return this->partitioners_.cbegin(); }  // NOLINT
  decltype(auto) cend() const { return this->partitioners_.cend(); }      // NOLINT
  decltype(auto) begin() const { return this->partitioners_.cbegin(); }   // NOLINT
  decltype(auto) end() const { return this->partitioners_.cend(); }       // NOLINT

  [[nodiscard]] decltype(auto) Front() { return this->partitioners_.front(); }
  [[nodiscard]] bool Empty() const { return this->partitioners_.empty(); }

  template <typename UpdatePositionOpT, typename OpDataT>
  void UpdatePositionBatch(Context const* ctx, std::int32_t batch_idx,
                           std::vector<bst_node_t> const& nidx,
                           std::vector<bst_node_t> const& left_nidx,
                           std::vector<bst_node_t> const& right_nidx,
                           std::vector<OpDataT> const& op_data, UpdatePositionOpT op) {
    auto& part = this->At(batch_idx);
    auto ridx_tmp = dh::ToSpan(this->ridx_tmp_).subspan(0, part->Size());
    part->UpdatePositionBatch(ctx, nidx, left_nidx, right_nidx, op_data, ridx_tmp, op);
  }
};
};  // namespace xgboost::tree
