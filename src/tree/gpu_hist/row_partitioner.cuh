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


const int kMaxBatch = 32;
template <typename OpDataT>
struct KernelArgs {
  Segment segments[kMaxBatch];
  OpDataT data[kMaxBatch];

  // Given a global thread idx, assign it to an item from one of the segments
  __device__ void AssignBatch(std::size_t idx, int &batch_idx, std::size_t &item_idx) const {
    std::size_t sum = 0;
    for (int i = 0; i < kMaxBatch; i++) {
      if (sum + segments[i].Size() > idx) {
        batch_idx = i;
        item_idx = (idx - sum) + segments[i].begin;
        break;
      }
      sum += segments[i].Size();
    }
  }
  std::size_t TotalRows() const {
    std::size_t total_rows = 0;
    for (auto segment : segments) {
      total_rows += segment.Size();
    }
    return total_rows;
  }
};

template <typename RowIndexT, typename OpT, typename OpDataT>
void GetLeftCounts(const KernelArgs<OpDataT>&args,common::Span<RowIndexT> ridx,
                   common::Span<unsigned long long int> d_left_counts, OpT op
                   ) {

  // Launch 1 thread for each row
  dh::LaunchN<1, 128>(args.TotalRows(), [=] __device__(std::size_t idx) {
    // Assign this thread to a row
    int batch_idx;
    std::size_t item_idx;
    args.AssignBatch(idx, batch_idx, item_idx);
    auto op_res = op(ridx[item_idx], args.data[batch_idx]);
    atomicAdd(&d_left_counts[batch_idx], op(ridx[item_idx], args.data[batch_idx]));
  });
}

struct IndexFlagTuple {
  size_t idx;
  bool flag;
  size_t flag_scan;
  int batch_idx;
};

struct IndexFlagOp {
  __device__ IndexFlagTuple operator()(const IndexFlagTuple& a, const IndexFlagTuple& b) const {
    if (a.batch_idx == b.batch_idx) {
      return {b.idx, b.flag, a.flag_scan + b.flag_scan, b.batch_idx};
    } else {
      return b;
    }
  }
};

template<typename OpDataT,typename OpT>
struct WriteResultsFunctor {
  KernelArgs<OpDataT> args;
  OpT op;
  common::Span<bst_uint> ridx_in;
  common::Span<bst_uint> ridx_out;
  common::Span<unsigned long long int> left_counts;

  __device__ IndexFlagTuple operator()(const IndexFlagTuple& x) {
    // the ex_scan_result represents how many rows have been assigned to left
    // node so far during scan.
    std::size_t scatter_address;
    if (x.flag) {
      scatter_address = args.segments[x.batch_idx].begin + x.flag_scan - 1;  // -1 because inclusive scan
    } else {
      // current number of rows belong to right node + total number of rows
      // belong to left node
      scatter_address  = (x.idx - x.flag_scan) + left_counts[x.batch_idx];
    }
    ridx_out[scatter_address] = ridx_in[x.idx];
    // Discard
    return {};
  }
};

template <typename RowIndexT, typename OpT, typename OpDataT>
void SortPositionBatch(const KernelArgs<OpDataT>& args, common::Span<RowIndexT> ridx,
                       common::Span<RowIndexT> ridx_tmp,
                       common::Span<unsigned long long int> left_counts, OpT op,
                       cudaStream_t stream) {
  WriteResultsFunctor<OpDataT,OpT> write_results{args,op,ridx, ridx_tmp, left_counts};
  auto discard_write_iterator =
      thrust::make_transform_output_iterator(dh::TypedDiscard<IndexFlagTuple>(), write_results);
  auto counting = thrust::make_counting_iterator(0llu);
  auto input_iterator =
      dh::MakeTransformIterator<IndexFlagTuple>(counting, [=] __device__(size_t idx) {
        int batch_idx;
        std::size_t item_idx;
        args.AssignBatch(idx, batch_idx, item_idx);
        auto go_left = op(ridx[item_idx], args.data[batch_idx]);
        return IndexFlagTuple{item_idx, go_left,go_left, batch_idx};
      });
  size_t temp_bytes = 0;
  cub::DeviceScan::InclusiveScan(nullptr, temp_bytes, input_iterator,
                                 discard_write_iterator, IndexFlagOp(),
                                 args.TotalRows(), stream);
  dh::TemporaryArray<int8_t> temp(temp_bytes);
  cub::DeviceScan::InclusiveScan(temp.data().get(), temp_bytes, input_iterator,
                                 discard_write_iterator, IndexFlagOp(), args.TotalRows(), stream);

  // copy active segments back to original buffer
  dh::LaunchN(args.TotalRows(), [=] __device__(std::size_t idx) {
    // Assign this thread to a row
    int batch_idx;
    std::size_t item_idx;
    args.AssignBatch(idx, batch_idx, item_idx);
    ridx[item_idx] = ridx_tmp[item_idx];
  });
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
  // Staging area for sorting ridx
  dh::TemporaryArray<RowIndexT> ridx_tmp_;
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
    CHECK_LE(nidx.size(), kMaxBatch);
    auto h_left_counts = pinned_.GetSpan<int64_t>(nidx.size(), 0);
    dh::TemporaryArray<unsigned long long int> d_left_counts(nidx.size(), 0);

    // Prepare kernel arguments
    KernelArgs<OpDataT> args;
    std::copy(op_data.begin(), op_data.end(), args.data);
    for (int i = 0; i < nidx.size(); i++) {
      args.segments[i] = ridx_segments_.at(nidx[i]);
    }
    GetLeftCounts(args, dh::ToSpan(ridx_), dh::ToSpan(d_left_counts), op);

    dh::safe_cuda(
        cudaMemcpyAsync(h_left_counts.data(), d_left_counts.data().get(),
                        sizeof(decltype(d_left_counts)::value_type) * d_left_counts.size(),
                        cudaMemcpyDefault, nullptr));

    SortPositionBatch(args, dh::ToSpan(ridx_), dh::ToSpan(ridx_tmp_), dh::ToSpan(d_left_counts), op,
                      nullptr);

    dh::safe_cuda(cudaDeviceSynchronize());

    // Update segments
    for (int i = 0; i < nidx.size(); i++) {
      auto segment=ridx_segments_.at(nidx[i]);
      auto left_count = h_left_counts[i];
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
