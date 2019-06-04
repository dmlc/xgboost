/*!
 * Copyright 2017-2019 XGBoost contributors
 */
#pragma once
#include "../../common/device_helpers.cuh"

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
  using TreePositionT = int;
  using RowIndexT = bst_uint;
  struct Segment;

 private:
  int device_idx;
  /*! \brief Range of rows for each node. */
  std::vector<Segment> ridx_segments;
  dh::caching_device_vector<RowIndexT> ridx_a;
  dh::caching_device_vector<RowIndexT> ridx_b;
  dh::caching_device_vector<TreePositionT> position_a;
  dh::caching_device_vector<TreePositionT> position_b;
  dh::DoubleBuffer<RowIndexT> ridx;
  dh::DoubleBuffer<TreePositionT> position;
  dh::caching_device_vector<int64_t>
      left_counts;  // Useful to keep a bunch of zeroed memory for sort position
  std::vector<cudaStream_t> streams;

 public:
  RowPartitioner(int device_idx, size_t num_rows);
  ~RowPartitioner();
  RowPartitioner(const RowPartitioner&) = delete;
  RowPartitioner& operator=(const RowPartitioner&) = delete;

  /**
   * \brief Gets the row indices of training instances in a given node.
   */
  common::Span<const RowIndexT> GetRows(TreePositionT nidx);

  /**
   * \brief Gets all training rows in the set.
   */
  common::Span<const RowIndexT> GetRows();

  /**
   * \brief Gets the tree position of all training instances.
   */
  common::Span<const TreePositionT> GetPosition();

  /**
   * \brief Convenience method for testing
   */
  std::vector<RowIndexT> GetRowsHost(TreePositionT nidx);

  /**
   * \brief Convenience method for testing
   */
  std::vector<TreePositionT> GetPositionHost();

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
  void UpdatePosition(TreePositionT nidx, TreePositionT left_nidx,
                      TreePositionT right_nidx, UpdatePositionOpT op) {
    dh::safe_cuda(cudaSetDevice(device_idx));
    Segment segment = ridx_segments.at(nidx);
    auto d_ridx = ridx.CurrentSpan();
    auto d_position = position.CurrentSpan();
    if (left_counts.size() <= nidx) {
      left_counts.resize((nidx * 2) + 1);
      thrust::fill(left_counts.begin(), left_counts.end(), 0);
    }
    int64_t* d_left_count = left_counts.data().get() + nidx;
    // Launch 1 thread for each row
    dh::LaunchN<1, 128>(device_idx, segment.Size(), [=] __device__(size_t idx) {
      idx += segment.begin;
      RowIndexT ridx = d_ridx[idx];
      // Missing value
      TreePositionT new_position = op(ridx);
      assert(new_position == left_nidx || new_position == right_nidx);
      AtomicIncrement(d_left_count, new_position == left_nidx);
      d_position[idx] = new_position;
    });
    // Overlap device to host memory copy (left_count) with sort
    int64_t left_count;
    dh::safe_cuda(cudaMemcpyAsync(&left_count, d_left_count, sizeof(int64_t),
                                  cudaMemcpyDeviceToHost, streams[0]));

    SortPositionAndCopy(segment, left_nidx, right_nidx, d_left_count,
                        streams[1]);

    dh::safe_cuda(cudaStreamSynchronize(streams[0]));
    CHECK_LE(left_count, segment.Size());
    CHECK_GE(left_count, 0);
    ridx_segments.resize(std::max(int(ridx_segments.size()),
                                  std::max(left_nidx, right_nidx) + 1));
    ridx_segments[left_nidx] =
        Segment(segment.begin, segment.begin + left_count);
    ridx_segments[right_nidx] =
        Segment(segment.begin + left_count, segment.end);
  }

  /**
   * \brief Finalise the position of all training instances after tree
   * construction is complete. Does not update any other meta information in
   * this data structure, so should only be used at the end of training.
   *
   * \param op          Device lambda. Should provide the row index  and current
   * position as an argument and return the new position for this training
   * instance.
   */
  template <typename FinalisePositionOpT>
  void FinalisePosition(FinalisePositionOpT op) {
    auto d_position = position.Current();
    const auto d_ridx = ridx.Current();
    dh::LaunchN(device_idx, position.Size(), [=] __device__(size_t idx) {
      auto position = d_position[idx];
      RowIndexT ridx = d_ridx[idx];
      d_position[idx] = op(ridx, position);
    });
  }

  /**
   * \brief Optimised routine for sorting key value pairs into left and right
   * segments. Based on a single pass of exclusive scan, uses iterators to
   * redirect inputs and outputs.
   */
  void SortPosition(common::Span<TreePositionT> position,
                    common::Span<TreePositionT> position_out,
                    common::Span<RowIndexT> ridx,
                    common::Span<RowIndexT> ridx_out, TreePositionT left_nidx,
                    TreePositionT right_nidx, int64_t* d_left_count,
                    cudaStream_t stream = nullptr);

  /*! \brief Sort row indices according to position. */
  void SortPositionAndCopy(const Segment& segment, TreePositionT left_nidx,
                           TreePositionT right_nidx, int64_t* d_left_count,
                           cudaStream_t stream);
  /** \brief Used to demarcate a contiguous set of row indices associated with
   * some tree node. */
  struct Segment {
    size_t begin;
    size_t end;

    Segment() : begin{0}, end{0} {}

    Segment(size_t begin, size_t end) : begin(begin), end(end) {
      CHECK_GE(end, begin);
    }
    size_t Size() const { return end - begin; }
  };
};
};  // namespace tree
};  // namespace xgboost
