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

constexpr int kUpdatePositionMaxBatch = 32;
template <typename OpDataT>
struct UpdatePositionBatchArgs {
  bst_node_t nidx_batch[kUpdatePositionMaxBatch];
  bst_node_t left_nidx_batch[kUpdatePositionMaxBatch];
  bst_node_t right_nidx_batch[kUpdatePositionMaxBatch];
  Segment segments_batch[kUpdatePositionMaxBatch];
  OpDataT data_batch[kUpdatePositionMaxBatch];
};

template <int kBlockSize, typename OpDataT, typename OpT>
__global__ void UpdatePositionBatchKernel(UpdatePositionBatchArgs<OpDataT> args,
                                          OpT op, common::Span<bst_uint> ridx,
                                          common::Span<bst_node_t> position,
                                          common::Span<int64_t> left_counts) {
  auto segment = args.segments_batch[blockIdx.x];
  auto data = args.data_batch[blockIdx.x];
  auto ridx_segment = ridx.subspan(segment.begin, segment.Size());
  auto position_segment = position.subspan(segment.begin, segment.Size());

  auto left_nidx = args.left_nidx_batch[blockIdx.x];
  auto left_count = dh::BlockPartition<kBlockSize>().Partition(
      ridx_segment.begin(), ridx_segment.end(), [=] __device__(auto e) { return op(e, data) == left_nidx; });

  if (threadIdx.x == 0) {
    left_counts[blockIdx.x] = left_count;
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
  std::vector<Segment> ridx_segments_;
  /*! \brief mapping for node id -> rows.
   * This looks like:
   * node id  |    1    |    2   |
   * rows idx | 3, 5, 1 | 13, 31 |
   */
  dh::TemporaryArray<RowIndexT> ridx_;
  /*! \brief mapping for row -> node id. */
  dh::TemporaryArray<bst_node_t> position_;
  dh::PinnedMemory pinned_;

 public:
  RowPartitioner(int device_idx, size_t num_rows);
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

  template <typename UpdatePositionOpT, typename OpDataT>
  void UpdatePositionBatch(const std::vector<bst_node_t>& nidx,
                           const std::vector<bst_node_t>& left_nidx,
                           const std::vector<bst_node_t>& right_nidx,
                           const std::vector<OpDataT>& op_data, UpdatePositionOpT op) {
    if (nidx.empty()) return;
    // Impose this limit because we are passing arguments for each node to the kernel by parameter
    // this avoids memcpy but we cannot pass arbitrary number of arguments
    CHECK_EQ(nidx.size(), left_nidx.size());
    CHECK_EQ(nidx.size(), right_nidx.size());
    CHECK_EQ(nidx.size(), op_data.size());
    CHECK_LE(nidx.size(), kUpdatePositionMaxBatch);
    auto left_counts = pinned_.GetSpan<int64_t>(nidx.size(), 0);

    // Prepare kernel arguments
    UpdatePositionBatchArgs<OpDataT> args;
    std::copy(nidx.begin(),nidx.end(),args.nidx_batch);
    std::copy(left_nidx.begin(),left_nidx.end(),args.left_nidx_batch);
    std::copy(right_nidx.begin(),right_nidx.end(),args.right_nidx_batch);
    std::copy(op_data.begin(),op_data.end(),args.data_batch);
    for(int i = 0; i < nidx.size(); i++){
      args.segments_batch[i]=ridx_segments_.at(nidx[i]);
    }

    // 1 block per node
    constexpr int kBlockSize = 512;
    UpdatePositionBatchKernel<kBlockSize><<<nidx.size(), kBlockSize>>>(
        args, op, dh::ToSpan(ridx_),
        dh::ToSpan(position_), left_counts);

    dh::safe_cuda(cudaDeviceSynchronize());

    // Update segments
    for (int i = 0; i < nidx.size(); i++) {
      auto segment=ridx_segments_.at(nidx[i]);
      auto left_count = left_counts[i];
      CHECK_LE(left_count, segment.Size());
      CHECK_GE(left_count, 0);
      ridx_segments_.resize(std::max(static_cast<bst_node_t>(ridx_segments_.size()),
                                     std::max(left_nidx[i], right_nidx[i]) + 1));
      ridx_segments_[left_nidx[i]] = Segment(segment.begin, segment.begin + left_count);
      ridx_segments_[right_nidx[i]] = Segment(segment.begin + left_count, segment.end);
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
  template <typename FinalisePositionOpT, typename Sampledp>
  void FinalisePosition(Context const* ctx, ObjInfo task,
                        HostDeviceVector<bst_node_t>* p_out_position, FinalisePositionOpT op,
                        Sampledp sampledp) {
    auto d_position = position_.data().get();
    const auto d_ridx = ridx_.data().get();
    if (!task.UpdateTreeLeaf()) {
      dh::LaunchN(position_.size(), [=] __device__(size_t idx) {
        auto position = d_position[idx];
        RowIndexT ridx = d_ridx[idx];
        bst_node_t new_position = op(ridx, position);
        if (new_position == kIgnoredTreePosition) {
          return;
        }
        d_position[idx] = new_position;
      });
      return;
    }

    p_out_position->SetDevice(ctx->gpu_id);
    p_out_position->Resize(position_.size());
    auto sorted_position = p_out_position->DevicePointer();
    dh::LaunchN(position_.size(), [=] __device__(size_t idx) {
      auto position = d_position[idx];
      RowIndexT ridx = d_ridx[idx];
      bst_node_t new_position = op(ridx, position);
      sorted_position[ridx] = sampledp(ridx) ? ~new_position : new_position;
      if (new_position == kIgnoredTreePosition) {
        return;
      }
      d_position[idx] = new_position;
    });
  }
};
};  // namespace tree
};  // namespace xgboost
