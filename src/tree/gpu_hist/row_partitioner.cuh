/*!
 * Copyright 2017-2019 XGBoost contributors
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include "xgboost/base.h"
#include "../../common/device_helpers.cuh"
#include "xgboost/generic_parameters.h"
#include "xgboost/task.h"
#include "xgboost/tree_model.h"

namespace xgboost {
namespace tree {

/*! \brief Count how many rows are assigned to left node. */
__forceinline__ __device__ void AtomicIncrement(int64_t* d_count, bool increment) {
#if __CUDACC_VER_MAJOR__ > 8
  int mask = __activemask();
  unsigned ballot = __ballot_sync(mask, increment);
  int leader = __ffs(mask) - 1;
  if (threadIdx.x % 32 == leader) {
    atomicAdd(reinterpret_cast<unsigned long long*>(d_count),    // NOLINT
              static_cast<unsigned long long>(__popc(ballot)));  // NOLINT
  }
#else
  unsigned ballot = __ballot(increment);
  if (threadIdx.x % 32 == 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(d_count),    // NOLINT
              static_cast<unsigned long long>(__popc(ballot)));  // NOLINT
  }
#endif
}

/** \brief Class responsible for tracking subsets of rows as we add splits and
 * partition training rows into different leaf nodes. */
class RowPartitioner {
 public:
  using RowIndexT = bst_uint;
  struct Segment;
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
  std::vector<Segment> ridx_segments_;
  dh::TemporaryArray<RowIndexT> ridx_a_;
  dh::TemporaryArray<RowIndexT> ridx_b_;
  dh::TemporaryArray<bst_node_t> position_a_;
  dh::TemporaryArray<bst_node_t> position_b_;
  /*! \brief mapping for node id -> rows.
   * This looks like:
   * node id  |    1    |    2   |
   * rows idx | 3, 5, 1 | 13, 31 |
   */
  dh::DoubleBuffer<RowIndexT> ridx_;
  /*! \brief mapping for row -> node id. */
  dh::DoubleBuffer<bst_node_t> position_;
  dh::caching_device_vector<int64_t>
      left_counts_;  // Useful to keep a bunch of zeroed memory for sort position
  std::vector<cudaStream_t> streams_;
  dh::PinnedMemory pinned_;

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
   * \brief Gets the tree position of all training instances.
   */
  common::Span<const bst_node_t> GetPosition();

  /**
   * \brief Convenience method for testing
   */
  std::vector<RowIndexT> GetRowsHost(bst_node_t nidx);

  /**
   * \brief Convenience method for testing
   */
  std::vector<bst_node_t> GetPositionHost();

  /**
   * \brief Updates the tree position for set of training instances being split
   * into left and right child nodes. Accepts a user-defined lambda specifying
   * which branch each training instance should go down.
   *
   * \tparam  UpdatePositionOpT
   * \param nidx        The index of the node being split.
   * \param left_nidx   The left child index.
   * \param right_nidx  The right child index.
   * \param op          Device lambda. Should provide the row index as an
   * argument and return the new position for this training instance.
   */
  template <typename UpdatePositionOpT>
  void UpdatePosition(bst_node_t nidx, bst_node_t left_nidx,
                      bst_node_t right_nidx, UpdatePositionOpT op) {
    Segment segment = ridx_segments_.at(nidx);  // rows belongs to node nidx
    auto d_ridx = ridx_.CurrentSpan();
    auto d_position = position_.CurrentSpan();
    if (left_counts_.size() <= nidx) {
      left_counts_.resize((nidx * 2) + 1);
      thrust::fill(left_counts_.begin(), left_counts_.end(), 0);
    }
    // Now we divide the row segment into left and right node.

    int64_t* d_left_count = left_counts_.data().get() + nidx;
    // Launch 1 thread for each row
    dh::LaunchN<1, 128>(segment.Size(), [segment, op, left_nidx, right_nidx, d_ridx, d_left_count,
                                         d_position] __device__(size_t idx) {
      // LaunchN starts from zero, so we restore the row index by adding segment.begin
      idx += segment.begin;
      RowIndexT ridx = d_ridx[idx];
      bst_node_t new_position = op(ridx);  // new node id
      KERNEL_CHECK(new_position == left_nidx || new_position == right_nidx);
      AtomicIncrement(d_left_count, new_position == left_nidx);
      d_position[idx] = new_position;
    });
    // Overlap device to host memory copy (left_count) with sort
    int64_t &left_count = pinned_.GetSpan<int64_t>(1)[0];
    dh::safe_cuda(cudaMemcpyAsync(&left_count, d_left_count, sizeof(int64_t),
                                  cudaMemcpyDeviceToHost, streams_[0]));

    SortPositionAndCopy(segment, left_nidx, right_nidx, d_left_count, streams_[1]);

    dh::safe_cuda(cudaStreamSynchronize(streams_[0]));
    CHECK_LE(left_count, segment.Size());
    CHECK_GE(left_count, 0);
    ridx_segments_.resize(std::max(static_cast<bst_node_t>(ridx_segments_.size()),
                                   std::max(left_nidx, right_nidx) + 1));
    ridx_segments_[left_nidx] =
        Segment(segment.begin, segment.begin + left_count);
    ridx_segments_[right_nidx] =
        Segment(segment.begin + left_count, segment.end);
  }

  /**
   * \brief Finalise the position of all training instances after tree
   * construction is complete. Does not update any other meta information in
   * this data structure, so should only be used at the end of training.
   *
   * \param p_out_row_indices Row partitions for each leaf.
   * \param op Device lambda. Should provide the row index and current position as an
   * argument and return the new position for this training instance.
   */
  template <typename FinalisePositionOpT, typename Sampledp>
  void FinalisePosition(Context const* ctx, RegTree const* p_tree, ObjInfo task,
                        std::vector<RowIndexCache>* p_out_row_indices, FinalisePositionOpT op,
                        Sampledp sampledp) {
    auto d_position = position_.Current();
    const auto d_ridx = ridx_.Current();
    auto sorted_position = position_.Other();
    if (!task.zero_hess) {
      dh::LaunchN(position_.Size(), [=] __device__(size_t idx) {
        auto position = d_position[idx];
        RowIndexT ridx = d_ridx[idx];
        bst_node_t new_position = op(ridx, position);
        if (new_position == kIgnoredTreePosition) {
          return;
        }
        d_position[idx] = new_position;
      });
      return;
    } else {
      dh::LaunchN(position_.Size(), [=] __device__(size_t idx) {
        auto position = d_position[idx];
        RowIndexT ridx = d_ridx[idx];
        bst_node_t new_position = op(ridx, position);
        if (sampledp(ridx)) {
          // push to the end
          sorted_position[ridx] = std::numeric_limits<bst_node_t>::max();
        } else {
          sorted_position[ridx] = new_position;
        }

        if (new_position == kIgnoredTreePosition) {
          return;
        }
        d_position[idx] = new_position;
      });
    }
    // copy position to buffer
    size_t n_samples = position_.Size();
    dh::XGBDeviceAllocator<char> alloc;
    auto& row_indices = p_out_row_indices->back();
    // sort row index according to node index
    row_indices.row_index.SetDevice(ctx->gpu_id);
    row_indices.row_index.Resize(ridx_.Size());
    dh::Iota(row_indices.row_index.DeviceSpan());
    thrust::stable_sort_by_key(thrust::cuda::par(alloc), sorted_position,
                               sorted_position + n_samples, row_indices.row_index.DevicePointer());

    size_t n_leaf = p_tree->GetNumLeaves();
    // +1 for subsample, which is set to a unique value in above kernel.
    size_t max_n_unique = n_leaf + 1;
    dh::caching_device_vector<size_t> unique_out(max_n_unique);
    dh::caching_device_vector<size_t> counts_out(max_n_unique);
    dh::TemporaryArray<size_t> num_runs_out(1);

    size_t nbytes;
    cub::DeviceRunLengthEncode::Encode(nullptr, nbytes, sorted_position, unique_out.data().get(),
                                       counts_out.data().get(), num_runs_out.data().get(),
                                       n_samples);
    dh::TemporaryArray<char> temp(nbytes);
    cub::DeviceRunLengthEncode::Encode(temp.data().get(), nbytes, sorted_position,
                                       unique_out.data().get(), counts_out.data().get(),
                                       num_runs_out.data().get(), n_samples);

    /**
     * copy node index (leaf index)
     */
    row_indices.node_idx.SetDevice(ctx->gpu_id);
    row_indices.node_idx.Resize(n_leaf);
    auto d_node_idx = row_indices.node_idx.DeviceSpan();
    // don't copy the sampled values
    thrust::copy(thrust::device, unique_out.begin(), unique_out.begin() + n_leaf,
                 dh::tbegin(d_node_idx));
    /**
     * copy node pointer
     */
    dh::XGBCachingDeviceAllocator<char> caching;
    row_indices.node_ptr.SetDevice(ctx->gpu_id);
    row_indices.node_ptr.Resize(n_leaf + 1, 0);
    auto d_node_ptr = row_indices.node_ptr.DeviceSpan();
    thrust::inclusive_scan(thrust::cuda::par(caching), counts_out.begin(),
                           counts_out.begin() + n_leaf, dh::tbegin(d_node_ptr) + 1);
  }

  /**
   * \brief Optimised routine for sorting key value pairs into left and right
   * segments. Based on a single pass of exclusive scan, uses iterators to
   * redirect inputs and outputs.
   */
  void SortPosition(common::Span<bst_node_t> position,
                    common::Span<bst_node_t> position_out,
                    common::Span<RowIndexT> ridx,
                    common::Span<RowIndexT> ridx_out, bst_node_t left_nidx,
                    bst_node_t right_nidx, int64_t* d_left_count,
                    cudaStream_t stream = nullptr);

  /*! \brief Sort row indices according to position. */
  void SortPositionAndCopy(const Segment& segment, bst_node_t left_nidx,
                           bst_node_t right_nidx, int64_t* d_left_count,
                           cudaStream_t stream);
  /** \brief Used to demarcate a contiguous set of row indices associated with
   * some tree node. */
  struct Segment {
    size_t begin { 0 };
    size_t end { 0 };

    Segment() = default;

    Segment(size_t begin, size_t end) : begin(begin), end(end) {
      CHECK_GE(end, begin);
    }
    size_t Size() const { return end - begin; }
  };
};
};  // namespace tree
};  // namespace xgboost
