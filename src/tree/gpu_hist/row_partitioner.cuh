/**
 * Copyright 2017-2025, XGBoost contributors
 */
#pragma once
#include <algorithm>  // for max, min
#include <array>      // for array
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t, uint32_t
#include <vector>     // for vector

#include "../../common/cuda_compat.cuh"     // for CUDA compatibility
#include "../../common/cuda_context.cuh"    // for CUDAContext
#include "../../common/device_helpers.cuh"  // for MakeTransformIterator
#include "xgboost/base.h"                   // for bst_idx_t
#include "xgboost/context.h"                // for Context
#include "xgboost/span.h"                   // for Span

namespace xgboost::tree {
namespace cuda_impl {
using RowIndexT = std::uint32_t;
// The rows of a node are split into tiles of this size, a block partitions one tile.
inline constexpr std::int32_t kPartitionTileRows = 1024;
inline constexpr std::int32_t kPartitionBlockThreads = 256;
inline constexpr std::int32_t kPartitionTileWords = kPartitionTileRows / 32;
// Each warp handles this many words of 32 rows.
inline constexpr std::int32_t kPartitionWarpWords =
    kPartitionTileWords / (kPartitionBlockThreads / 32);
}  // namespace cuda_impl

/**
 * @brief The two row buffers of a partitioner.
 *
 * Partitioning a node moves its rows from its buffer to the other one, so the rows of a node at
 * depth d are in buffer d % 2. A node's segment is the same range in both buffers.
 */
template <typename T>
struct RowBuffers {
  common::Span<T> buffers[2];

  [[nodiscard]] XGBOOST_DEVICE common::Span<T> operator[](std::int32_t i) const {
    // A select instead of indexing, so the array is not placed in local memory.
    return i == 0 ? buffers[0] : buffers[1];
  }
};

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
  /** @brief The number of tiles before this node in the node batch. */
  cuda_impl::RowIndexT tile_offset{0};
  /** @brief The buffer holding the rows of this node, see @ref RowBuffers. */
  std::int32_t buffer{0};
};

/**
 * @brief State of the partition kernel, zero between launches.
 *
 * The words are the number of finished blocks, followed by the number of rows written to the
 * left and to the right of each node in the batch.
 */
struct PartitionState {
  std::uint32_t* words;

  [[nodiscard]] static std::size_t Size(std::size_t n_nodes) { return 1 + 2 * n_nodes; }
  [[nodiscard]] XGBOOST_DEV_INLINE std::uint32_t* NDone() const { return words; }
  [[nodiscard]] XGBOOST_DEV_INLINE std::uint32_t* NLeft(std::int32_t i) const {
    return words + 1 + 2 * i;
  }
  [[nodiscard]] XGBOOST_DEV_INLINE std::uint32_t* NRight(std::int32_t i) const {
    return words + 2 + 2 * i;
  }
};

/**
 * @brief Partition the rows of a batch of nodes in a single pass.
 *
 * Each block reads a tile of a node's rows from the node's buffer. It reserves space for its left
 * and right rows with an atomic for each side, and writes them to the other buffer: left rows
 * from the beginning of the segment, right rows from the end. The order of rows within a node
 * depends on the order of the blocks.
 *
 * The last block writes the number of rows going left for each node, and resets the state.
 */
template <typename OpT, typename OpDataT>
__global__ __launch_bounds__(cuda_impl::kPartitionBlockThreads) void PartitionKernel(
    dh::LDGIterator<PerNodeData<OpDataT>> batch_info, std::int32_t n_nodes, std::uint32_t n_tiles,
    RowBuffers<cuda_impl::RowIndexT> ridx, PartitionState state,
    cuda_impl::RowIndexT* d_counts, OpT op) {
  using cuda_impl::kPartitionTileRows;
  using cuda_impl::kPartitionTileWords;
  using cuda_impl::kPartitionWarpWords;
  using RowIndexT = cuda_impl::RowIndexT;
  __shared__ std::uint32_t word_offset[kPartitionTileWords];
  __shared__ std::uint32_t base[2];
  __shared__ bool is_last;

  auto const warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  auto const lanemask_lt = (1u << lane) - 1u;

  std::uint32_t const t = blockIdx.x;
  if (t < n_tiles) {
    // Find the node of this tile, empty nodes are skipped.
    auto offsets = dh::MakeTransformIterator<RowIndexT>(
        dh::make_counting_iterator(0), [=](std::int32_t i) { return batch_info[i].tile_offset; });
    auto i = static_cast<std::int32_t>(dh::SegmentId(offsets, offsets + n_nodes, t));
    auto const node = batch_info[i];
    auto const& segment = node.segment;
    RowIndexT const* in = ridx[node.buffer].data();
    RowIndexT* out = ridx[1 - node.buffer].data();
    auto tile = segment.begin + (t - node.tile_offset) * kPartitionTileRows;
    auto n_rows = std::min(segment.end - tile, static_cast<RowIndexT>(kPartitionTileRows));

    RowIndexT rows[kPartitionWarpWords];
    bool go_left[kPartitionWarpWords];
#pragma unroll
    for (std::int32_t j = 0; j < kPartitionWarpWords; ++j) {
      auto local = (warp * kPartitionWarpWords + j) * 32 + lane;
      bool valid = local < n_rows;
      rows[j] = valid ? in[tile + local] : 0;
      go_left[j] = valid && op(rows[j], node.data);
    }
    // Rank the left rows within the tile.
    std::uint32_t bits[kPartitionWarpWords];
#pragma unroll
    for (std::int32_t j = 0; j < kPartitionWarpWords; ++j) {
      bits[j] = __ballot_sync(0xffffffff, go_left[j]);
      if (lane == 0) {
        word_offset[warp * kPartitionWarpWords + j] = __popc(bits[j]);
      }
    }
    __syncthreads();
    if (warp == 0) {
      auto v = word_offset[lane];
      auto inclusive = v;
#pragma unroll
      for (std::int32_t d = 1; d < 32; d *= 2) {
        auto n = __shfl_up_sync(0xffffffff, inclusive, d);
        if (lane >= d) {
          inclusive += n;
        }
      }
      word_offset[lane] = inclusive - v;
      if (lane == 31) {
        // Reserve space on both sides of the node.
        base[0] = atomicAdd(state.NLeft(i), inclusive);
        base[1] = atomicAdd(state.NRight(i), n_rows - inclusive);
      }
    }
    __syncthreads();
#pragma unroll
    for (std::int32_t j = 0; j < kPartitionWarpWords; ++j) {
      auto local = (warp * kPartitionWarpWords + j) * 32 + lane;
      if (local < n_rows) {
        auto n_left_before =
            word_offset[warp * kPartitionWarpWords + j] + __popc(bits[j] & lanemask_lt);
        if (go_left[j]) {
          out[segment.begin + base[0] + n_left_before] = rows[j];
        } else {
          out[segment.end - 1 - (base[1] + local - n_left_before)] = rows[j];
        }
      }
    }
  }

  // The last block publishes the counts, and resets the state for the next launch.
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    is_last = atomicAdd(state.NDone(), 1u) == gridDim.x - 1;
  }
  __syncthreads();
  if (is_last) {
    __threadfence();
    for (std::int32_t i = threadIdx.x; i < n_nodes; i += blockDim.x) {
      d_counts[i] = atomicExch(state.NLeft(i), 0u);
      *state.NRight(i) = 0;
    }
    if (threadIdx.x == 0) {
      *state.NDone() = 0;
    }
  }
}

/**
 * @brief Partition the rows of a batch of nodes, from the buffer of each node to the other one.
 *
 * @param d_batch_info Node data, with the size of the input number of nodes.
 * @param n_tiles      The number of tiles, see @ref PerNodeData::tile_offset.
 * @param d_counts     Output, the number of rows going left for each node.
 * @param tmp          Storage for the state between launches.
 */
template <typename OpT, typename OpDataT>
void SortPositionBatch(Context const* ctx, common::Span<const PerNodeData<OpDataT>> d_batch_info,
                       RowBuffers<cuda_impl::RowIndexT> ridx,
                       common::Span<cuda_impl::RowIndexT> d_counts, std::size_t n_tiles, OpT op,
                       dh::DeviceUVector<std::uint32_t>* tmp) {
  auto stream = ctx->CUDACtx()->Stream();
  auto n_nodes = d_batch_info.size();
  if (tmp->size() < PartitionState::Size(n_nodes)) {
    tmp->resize(PartitionState::Size(n_nodes));
    dh::safe_cuda(cudaMemsetAsync(tmp->data(), 0, tmp->size() * sizeof(std::uint32_t), stream));
  }
  // At least one block, to write the counts.
  auto n_blocks = static_cast<std::uint32_t>(std::max<std::size_t>(n_tiles, 1));
  dh::LaunchKernel{n_blocks, cuda_impl::kPartitionBlockThreads, 0, stream}(
      PartitionKernel<OpT, OpDataT>, dh::LDGIterator<PerNodeData<OpDataT>>{d_batch_info.data()},
      static_cast<std::int32_t>(n_nodes), static_cast<std::uint32_t>(n_tiles), ridx,
      PartitionState{tmp->data()}, d_counts.data(), op);
}

struct NodePositionInfo {
  Segment segment;
  bst_node_t left_child = -1;
  bst_node_t right_child = -1;
  // The buffer holding the rows of this node, see @ref RowBuffers.
  std::int32_t buffer = 0;
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
    RowBuffers<const cuda_impl::RowIndexT> ridx, common::Span<bst_node_t> d_out_position,
    OpT op) {
  for (auto idx : dh::GridStrideRange<std::size_t>(0, ridx[0].size())) {
    auto position = GetPositionFromSegments(idx, d_node_info.data());
    // Read from the buffer holding the rows of the leaf.
    auto global_ridx = ridx[d_node_info[position].buffer][idx];
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
  std::array<dh::DeviceUVector<RowIndexT>, 2> ridx_;
  // State of the partition kernel.
  dh::DeviceUVector<std::uint32_t> tmp_;

  [[nodiscard]] RowBuffers<RowIndexT> Buffers() {
    return {{dh::ToSpan(ridx_[0]), dh::ToSpan(ridx_[1])}};
  }
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
   * @brief Gets both row buffers, see @ref RowBuffers.
   */
  [[nodiscard]] RowBuffers<const RowIndexT> GetBuffers() const {
    return {{dh::ToSpan(ridx_[0]), dh::ToSpan(ridx_[1])}};
  }
  /**
   * @brief Get the number of rows in this partitioner.
   */
  std::size_t Size() const { return ridx_[0].size(); }

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
                           std::vector<OpDataT> const& op_data, UpdatePositionOpT op) {
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

    std::size_t n_tiles = 0;
    for (std::size_t i = 0; i < nidx.size(); i++) {
      auto const& node = ridx_segments_.at(nidx[i]);
      h_batch_info[i] = {node.segment, op_data[i], static_cast<RowIndexT>(n_tiles), node.buffer};
      n_tiles += common::DivRoundUp(node.segment.Size(), cuda_impl::kPartitionTileRows);
    }
    dh::safe_cuda(cudaMemcpyAsync(d_batch_info.data().get(), h_batch_info.data(),
                                  h_batch_info.size_bytes(), cudaMemcpyDefault,
                                  ctx->CUDACtx()->Stream()));
    // Temporary arrays
    auto h_counts = pinned_.GetSpan<RowIndexT>(nidx.size());
    // Written for every node by the kernel.
    dh::TemporaryArray<RowIndexT> d_counts(nidx.size());

    // Partition the rows of all nodes according to the operator
    SortPositionBatch<UpdatePositionOpT, OpDataT>(ctx, dh::ToSpan(d_batch_info), this->Buffers(),
                                                  dh::ToSpan(d_counts), n_tiles, op, &this->tmp_);

    dh::safe_cuda(cudaMemcpyAsync(h_counts.data(), d_counts.data().get(), h_counts.size_bytes(),
                                  cudaMemcpyDefault, ctx->CUDACtx()->Stream()));
    // TODO(Rory): this synchronisation hurts performance a lot
    // Future optimisation should find a way to skip this
    ctx->CUDACtx()->Stream().Sync();

    // Update segments
    for (std::size_t i = 0; i < nidx.size(); i++) {
      auto segment = ridx_segments_.at(nidx[i]).segment;
      auto buffer = ridx_segments_.at(nidx[i]).buffer;
      auto left_count = h_counts[i];
      CHECK_LE(left_count, segment.Size());
      ridx_segments_.resize(std::max(static_cast<bst_node_t>(ridx_segments_.size()),
                                     std::max(left_nidx[i], right_nidx[i]) + 1));
      ridx_segments_[nidx[i]] = NodePositionInfo{segment, left_nidx[i], right_nidx[i], buffer};
      // The children are in the other buffer.
      ridx_segments_[left_nidx[i]] =
          NodePositionInfo{Segment{segment.begin, segment.begin + left_count}, -1, -1, 1 - buffer};
      ridx_segments_[right_nidx[i]] =
          NodePositionInfo{Segment{segment.begin + left_count, segment.end}, -1, -1, 1 - buffer};
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
        xgboost::common::DivRoundUp(this->Size(), kBlockSize * kItemsThread);
    dh::LaunchKernel{grid_size, kBlockSize, 0, ctx->CUDACtx()->Stream()}(
        FinalisePositionKernel<kBlockSize, FinalisePositionOpT>, dh::ToSpan(d_node_info_storage),
        base_ridx, this->GetBuffers(), d_out_position, op);
  }
};

// Partitioner for all batches, used for external memory training.
class RowPartitionerBatches {
 private:
  // Partitioners for each batch.
  std::vector<std::unique_ptr<RowPartitioner>> partitioners_;

 public:
  void Reset(Context const* ctx, std::vector<bst_idx_t> const& batch_ptr) {
    CHECK_GE(batch_ptr.size(), 2);
    std::size_t n_batches = batch_ptr.size() - 1;
    if (partitioners_.size() != n_batches) {
      partitioners_.clear();
    }

    for (std::size_t k = 0; k < n_batches; ++k) {
      if (partitioners_.size() != n_batches) {
        // First run.
        partitioners_.emplace_back(std::make_unique<RowPartitioner>());
      }
      auto base_ridx = batch_ptr[k];
      auto n_samples = batch_ptr.at(k + 1) - base_ridx;
      partitioners_[k]->Reset(ctx, n_samples, base_ridx);
      CHECK_LE(n_samples, std::numeric_limits<cuda_impl::RowIndexT>::max());
    }
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
    this->At(batch_idx)->UpdatePositionBatch(ctx, nidx, left_nidx, right_nidx, op_data, op);
  }
};
};  // namespace xgboost::tree
