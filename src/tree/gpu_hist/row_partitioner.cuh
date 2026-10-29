/**
 * Copyright 2017-2025, XGBoost contributors
 */
#pragma once
#include <cub/version.cuh>  // for CUB_VERSION
#if CUB_VERSION >= 300000
#include <cuda/iterator>  // for tabulate_output_iterator
#else
#include <thrust/iterator/tabulate_output_iterator.h>  // for tabulate_output_iterator
#endif

#include <algorithm>        // for max
#include <cstddef>          // for size_t
#include <cstdint>          // for int32_t, uint32_t
#include <cub/device/device_scan.cuh>  // for DeviceScan
#include <cuda/functional>  // for proclaim_return_type
#include <vector>           // for vector

#include "../../common/cuda_compat.cuh"     // for CUDA compatibility
#include "../../common/cuda_context.cuh"    // for CUDAContext
#include "../../common/device_helpers.cuh"  // for MakeTransformIterator
#include "xgboost/base.h"                   // for bst_idx_t
#include "xgboost/context.h"                // for Context
#include "xgboost/span.h"                   // for Span

namespace xgboost::tree {
namespace cuda_impl {
using RowIndexT = std::uint32_t;

#if CUB_VERSION >= 300000
template <typename Fn>
using TabulateOutputIterator = cuda::tabulate_output_iterator<Fn>;
#else
template <typename Fn>
using TabulateOutputIterator = thrust::tabulate_output_iterator<Fn>;
#endif  // CUB_VERSION >= 300000

/**
 * @brief Output iterator calling `fn(index, value)` for each output.
 *
 * The tabulate output iterator declares a `void` value type, while the scan tuning in cub
 * requires a complete output value type.
 */
template <typename Fn, typename T>
class TypedTabulateOutput : public TabulateOutputIterator<Fn> {
 public:
  using value_type = T;  // NOLINT
  XGBOOST_DEVICE explicit TypedTabulateOutput(Fn fn) : TabulateOutputIterator<Fn>{fn} {}
};
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
  [[nodiscard]] XGBOOST_DEVICE bst_idx_t Size() const { return end - begin; }
};

template <typename OpDataT>
struct PerNodeData {
  Segment segment;
  OpDataT data;
  /** @brief The number of rows before this node in the node batch. */
  cuda_impl::RowIndexT batch_offset{0};
};

/**
 * @param n_nodes The number of nodes in this node batch.
 * @param global_thread_idx In practice, the row index within the total number of rows for
 *        this node batch.
 * @param batch_idx The nidx within this node batch (not the actual node index in a tree).
 * @param item_idx The resulting global row index (without accounting for base_rowid). This maps the
 *        row index within the node batch back to the global row index.
 */
template <typename T>
XGBOOST_DEV_INLINE void AssignBatch(dh::LDGIterator<T> const& batch_info_iter, std::int32_t n_nodes,
                                    cuda_impl::RowIndexT global_thread_idx, int* batch_idx,
                                    cuda_impl::RowIndexT* item_idx) {
  // Binary search for the nidx in batch, empty nodes are skipped.
  auto offsets = dh::MakeTransformIterator<cuda_impl::RowIndexT>(
      dh::make_counting_iterator(0),
      [=](std::int32_t i) { return batch_info_iter[i].batch_offset; });
  *batch_idx =
      static_cast<std::int32_t>(dh::SegmentId(offsets, offsets + n_nodes, global_thread_idx));
  auto node = batch_info_iter[*batch_idx];
  // the beginning of the segment plus the offset into that segment
  *item_idx = (global_thread_idx - node.batch_offset) + node.segment.begin;
}

/**
 * @param total_rows The total number of rows for this batch of nodes.
 */
template <int kBlockSize, typename OpDataT>
__global__ __launch_bounds__(kBlockSize) void SortPositionCopyKernel(
    dh::LDGIterator<PerNodeData<OpDataT>> batch_info_iter, std::int32_t n_nodes,
    common::Span<cuda_impl::RowIndexT> d_ridx,
    common::Span<cuda_impl::RowIndexT const> const ridx_tmp, bst_idx_t total_rows) {
  // The loop counter stays 64-bit so the stride can't wrap around near the 32-bit row limit.
  for (auto idx : dh::GridStrideRange<std::size_t>(0, total_rows)) {
    std::int32_t batch_idx;  // unused
    cuda_impl::RowIndexT item_idx;
    AssignBatch(batch_info_iter, n_nodes, static_cast<cuda_impl::RowIndexT>(idx), &batch_idx,
                &item_idx);
    d_ridx[item_idx] = ridx_tmp[item_idx];
  }
}

/**
 * @brief We scan over this value to find how to partition the inputs according to the flag.
 *
 * It packs the node in the batch (high 31 bits), whether the row goes left (1 bit), and the
 * number of rows going left so far (low 32 bits) into one word. cub can then keep the status and
 * the value of each tile in a single word.
 */
struct IndexFlag {
  static constexpr std::uint32_t kFlagShift = 32;
  static constexpr std::uint32_t kBatchShift = 33;

  [[nodiscard]] XGBOOST_DEV_INLINE static std::uint64_t Make(std::int32_t batch_idx, bool flag) {
    return (static_cast<std::uint64_t>(batch_idx) << kBatchShift) |
           (static_cast<std::uint64_t>(flag) << kFlagShift) | static_cast<std::uint64_t>(flag);
  }
  [[nodiscard]] XGBOOST_DEV_INLINE static std::int32_t BatchIdx(std::uint64_t x) {
    return static_cast<std::int32_t>(x >> kBatchShift);
  }
  [[nodiscard]] XGBOOST_DEV_INLINE static bool Flag(std::uint64_t x) {
    return (x >> kFlagShift) & 1;
  }
  [[nodiscard]] XGBOOST_DEV_INLINE static cuda_impl::RowIndexT FlagScan(std::uint64_t x) {
    return static_cast<cuda_impl::RowIndexT>(x);
  }
};

struct IndexFlagOp {
  __device__ std::uint64_t operator()(std::uint64_t a, std::uint64_t b) const {
    // Segmented scan - resets if we cross batch boundaries
    if (IndexFlag::BatchIdx(a) == IndexFlag::BatchIdx(b)) {
      // Accumulate the flags. The count is bounded by the number of rows, it doesn't carry into
      // the flag.
      return b + IndexFlag::FlagScan(a);
    }
    return b;
  }
};

// Scatter from `ridx_in` to `ridx_out`.
template <typename OpDataT>
struct WriteResultsFunctor {
  dh::LDGIterator<PerNodeData<OpDataT>> batch_info;
  cuda_impl::RowIndexT const* ridx_in;
  cuda_impl::RowIndexT* ridx_out;
  cuda_impl::RowIndexT* counts;

  // `idx` is the position of the scanned value, the row index within the node batch.
  __device__ void operator()(std::ptrdiff_t idx, std::uint64_t x) const {
    auto batch_idx = IndexFlag::BatchIdx(x);
    auto flag_scan = IndexFlag::FlagScan(x);
    // Get the node that this row belongs to.
    auto node = batch_info[batch_idx];
    auto const& segment = node.segment;
    // The location of the item we are working on in ridx_
    auto item_idx = static_cast<cuda_impl::RowIndexT>(idx) - node.batch_offset + segment.begin;
    cuda_impl::RowIndexT scatter_address;
    if (IndexFlag::Flag(x)) {
      // Go left.
      cuda_impl::RowIndexT num_previous_flagged = flag_scan - 1;  // -1 because inclusive scan
      scatter_address = segment.begin + num_previous_flagged;
    } else {
      cuda_impl::RowIndexT num_previous_unflagged = (item_idx - segment.begin) - flag_scan;
      scatter_address = segment.end - num_previous_unflagged - 1;
    }
    ridx_out[scatter_address] = ridx_in[item_idx];

    if (item_idx == (segment.end - 1)) {
      // Write out counts
      counts[batch_idx] = flag_scan;
    }
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
  auto n_nodes = static_cast<std::int32_t>(d_batch_info.size());
  WriteResultsFunctor<OpDataT> write_results{batch_info_itr, ridx.data(), ridx_tmp.data(),
                                             d_counts.data()};

  cuda_impl::TypedTabulateOutput<WriteResultsFunctor<OpDataT>, std::uint64_t> write_iterator{
      write_results};
  // A partitioner holds fewer than 2^32 rows, use 32-bit indices for the scan.
  CHECK_LE(total_rows, std::numeric_limits<cuda_impl::RowIndexT>::max());
  auto counting = dh::make_counting_iterator(static_cast<cuda_impl::RowIndexT>(0));
  auto input_iterator = dh::MakeTransformIterator<std::uint64_t>(
      counting, cuda::proclaim_return_type<std::uint64_t>([=] __device__(cuda_impl::RowIndexT idx) {
        std::int32_t nidx_in_batch;
        cuda_impl::RowIndexT item_idx;
        AssignBatch(batch_info_itr, n_nodes, idx, &nidx_in_batch, &item_idx);
        auto go_left = op(ridx[item_idx], batch_info_itr[nidx_in_batch].data);
        return IndexFlag::Make(nidx_in_batch, go_left);
      }));
  std::size_t n_bytes = 0;
  auto scan = [&](void* storage) {
    return cub::DeviceScan::InclusiveScan(
        storage, n_bytes, input_iterator, write_iterator, IndexFlagOp{},
        static_cast<cuda_impl::RowIndexT>(total_rows), ctx->CUDACtx()->Stream());
  };
  if (tmp->empty()) {
    // The size of temporary storage is calculated based on the total number of
    // rows. Since the root node has all the rows, subsequence allocatioin must be smaller
    // than the root node. As a result, we can calculate this once and reuse it throughout
    // the iteration.
    dh::safe_cuda(scan(nullptr));
    tmp->resize(n_bytes);
  }
  n_bytes = tmp->size();
  dh::safe_cuda(scan(tmp->data()));

  constexpr int kBlockSize = 256;

  // Value found by experimentation
  const int kItemsThread = 12;
  std::uint32_t const kGridSize =
      xgboost::common::DivRoundUp(total_rows, kBlockSize * kItemsThread);
  dh::LaunchKernel{kGridSize, kBlockSize, 0, ctx->CUDACtx()->Stream()}(
      SortPositionCopyKernel<kBlockSize, OpDataT>, batch_info_itr, n_nodes, ridx, ridx_tmp,
      total_rows);
}

struct NodePositionInfo {
  Segment segment;
  bst_node_t left_child = -1;
  bst_node_t right_child = -1;
  [[nodiscard]] XGBOOST_DEVICE bool IsLeaf() const { return left_child == -1; }
};

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
  /**
   * @brief mapping for node id -> rows.
   *
   * This looks like:
   * node id  |    1    |    2   |
   * rows idx | 3, 5, 1 | 13, 31 |
   */
  dh::DeviceUVector<RowIndexT> ridx_;
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
    common::Span<PerNodeData<OpDataT>> h_batch_info =
        pinned2_.GetSpan<PerNodeData<OpDataT>>(nidx.size());
    dh::TemporaryArray<PerNodeData<OpDataT>> d_batch_info(nidx.size());

    RowIndexT total_rows = 0;
    for (std::size_t i = 0; i < nidx.size(); i++) {
      auto const& segment = ridx_segments_.at(nidx[i]).segment;
      h_batch_info[i] = {segment, op_data[i], total_rows};
      total_rows += segment.Size();
    }
    dh::safe_cuda(cudaMemcpyAsync(d_batch_info.data().get(), h_batch_info.data(),
                                  h_batch_info.size_bytes(), cudaMemcpyDefault,
                                  ctx->CUDACtx()->Stream()));
    // Temporary arrays
    auto h_counts = pinned_.GetSpan<RowIndexT>(nidx.size());
    // Must initialize with 0 as 0 count is not written in the kernel.
    dh::TemporaryArray<RowIndexT> d_counts(nidx.size(), 0);
    CHECK_EQ(ridx_tmp.size(), this->Size());

    // Partition the rows of all nodes according to the operator
    SortPositionBatch<UpdatePositionOpT, OpDataT>(
        ctx, dh::ToSpan(d_batch_info), dh::ToSpan(this->ridx_), ridx_tmp, dh::ToSpan(d_counts),
        total_rows, op, &this->tmp_);

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
   * @param p_out_position Node index for each row in this batch.
   * @param op Device lambda. Receives the global row index and current position, and returns the
   *           new position for this training instance.
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
