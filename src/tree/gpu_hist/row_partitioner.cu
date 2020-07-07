/*!
 * Copyright 2017-2019 XGBoost contributors
 */
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sequence.h>
#include <vector>
#include "../../common/device_helpers.cuh"
#include "row_partitioner.cuh"

namespace xgboost {
namespace tree {

struct IndicateLeftTransform {
  bst_node_t left_nidx;
  explicit IndicateLeftTransform(bst_node_t left_nidx) : left_nidx(left_nidx) {}
  __host__ __device__ __forceinline__ size_t
  operator()(const bst_node_t& x) const {
    return x == left_nidx ? 1 : 0;
  }
};

struct IndexFlagTuple {
  size_t idx;
  size_t flag;
};

struct IndexFlagOp {
  __device__ IndexFlagTuple operator()(const IndexFlagTuple& a,
                                       const IndexFlagTuple& b) const {
    return {b.idx, a.flag + b.flag};
  }
};

struct WriteResultsFunctor {
  bst_node_t left_nidx;
  common::Span<bst_node_t> position_in;
  common::Span<bst_node_t> position_out;
  common::Span<RowPartitioner::RowIndexT> ridx_in;
  common::Span<RowPartitioner::RowIndexT> ridx_out;
  int64_t* d_left_count;

  __device__ IndexFlagTuple operator()(const IndexFlagTuple& x) {
    // the ex_scan_result represents how many rows have been assigned to left
    // node so far during scan.
    int scatter_address;
    if (position_in[x.idx] == left_nidx) {
      scatter_address = x.flag - 1;  // -1 because inclusive scan
    } else {
      // current number of rows belong to right node + total number of rows
      // belong to left node
      scatter_address = (x.idx - x.flag) + *d_left_count;
    }
    // copy the node id to output
    position_out[scatter_address] = position_in[x.idx];
    ridx_out[scatter_address] = ridx_in[x.idx];

    // Discard
    return {};
  }
};

// Change the value type of thrust discard iterator so we can use it with cub
class DiscardOverload : public thrust::discard_iterator<IndexFlagTuple> {
 public:
  using value_type = IndexFlagTuple;  // NOLINT
};

// Implement partitioning via single scan operation using transform output to
// write the result
void RowPartitioner::SortPosition(common::Span<bst_node_t> position,
                                  common::Span<bst_node_t> position_out,
                                  common::Span<RowIndexT> ridx,
                                  common::Span<RowIndexT> ridx_out,
                                  bst_node_t left_nidx, bst_node_t right_nidx,
                                  int64_t* d_left_count, cudaStream_t stream) {
  WriteResultsFunctor write_results{left_nidx, position, position_out,
                                    ridx,      ridx_out, d_left_count};
  auto discard_write_iterator =
      thrust::make_transform_output_iterator(DiscardOverload(), write_results);
  auto counting = thrust::make_counting_iterator(0llu);
  auto input_iterator = dh::MakeTransformIterator<IndexFlagTuple>(
      counting, [=] __device__(size_t idx) {
        return IndexFlagTuple{idx, position[idx] == left_nidx};
      });
  size_t temp_bytes = 0;
  cub::DeviceScan::InclusiveScan(nullptr, temp_bytes, input_iterator,
                                 discard_write_iterator, IndexFlagOp(),
                                 position.size(), stream);
  dh::TemporaryArray<int8_t> temp(temp_bytes);
  cub::DeviceScan::InclusiveScan(temp.data().get(), temp_bytes, input_iterator,
                                 discard_write_iterator, IndexFlagOp(),
                                 position.size(), stream);
}

void Reset(int device_idx, common::Span<RowPartitioner::RowIndexT> ridx,
           common::Span<bst_node_t> position) {
  CHECK_EQ(ridx.size(), position.size());
  dh::LaunchN(device_idx, ridx.size(), [=] __device__(size_t idx) {
    ridx[idx] = idx;
    position[idx] = 0;
  });
}

RowPartitioner::RowPartitioner(int device_idx, size_t num_rows)
    : device_idx_(device_idx), ridx_a_(num_rows), position_a_(num_rows) {
  dh::safe_cuda(cudaSetDevice(device_idx_));
  Reset(device_idx, dh::ToSpan(ridx_a_), dh::ToSpan(position_a_));
  left_counts_.resize(256);
  thrust::fill(left_counts_.begin(), left_counts_.end(), 0);
  streams_.resize(2);
  ridx_segments_.emplace_back(Segment(0, num_rows));
  for (auto& stream : streams_) {
    dh::safe_cuda(cudaStreamCreate(&stream));
  }
}
RowPartitioner::~RowPartitioner() {
  dh::safe_cuda(cudaSetDevice(device_idx_));
  for (auto& stream : streams_) {
    dh::safe_cuda(cudaStreamDestroy(stream));
  }
}

common::Span<const RowPartitioner::RowIndexT> RowPartitioner::GetRows(
    bst_node_t nidx) {
  auto segment = ridx_segments_.at(nidx);
  // Return empty span here as a valid result
  // Will error if we try to construct a span from a pointer with size 0
  if (segment.Size() == 0) {
    return common::Span<const RowPartitioner::RowIndexT>();
  }
  return dh::ToSpan(ridx_a_).subspan(segment.begin, segment.Size());
}

common::Span<const RowPartitioner::RowIndexT> RowPartitioner::GetRows() {
  return dh::ToSpan(ridx_a_);
}

common::Span<const bst_node_t> RowPartitioner::GetPosition() {
  return dh::ToSpan(position_a_);
}
std::vector<RowPartitioner::RowIndexT> RowPartitioner::GetRowsHost(
    bst_node_t nidx) {
  auto span = GetRows(nidx);
  std::vector<RowIndexT> rows(span.size());
  dh::CopyDeviceSpanToVector(&rows, span);
  return rows;
}

std::vector<bst_node_t> RowPartitioner::GetPositionHost() {
  auto span = GetPosition();
  std::vector<bst_node_t> position(span.size());
  dh::CopyDeviceSpanToVector(&position, span);
  return position;
}

void RowPartitioner::SortPositionAndCopy(const Segment& segment,
                                         bst_node_t left_nidx,
                                         bst_node_t right_nidx,
                                         int64_t* d_left_count,
                                         cudaStream_t stream) {
  dh::TemporaryArray<bst_node_t> position_temp(position_a_.size());
  dh::TemporaryArray<RowIndexT> ridx_temp(ridx_a_.size());
  SortPosition(
      // position_in
      common::Span<bst_node_t>(position_a_.data().get() + segment.begin,
                               segment.Size()),
      // position_out
      common::Span<bst_node_t>(position_temp.data().get() + segment.begin,
                               segment.Size()),
      // row index in
      common::Span<RowIndexT>(ridx_a_.data().get() + segment.begin, segment.Size()),
      // row index out
      common::Span<RowIndexT>(ridx_temp.data().get() + segment.begin, segment.Size()),
      left_nidx, right_nidx, d_left_count, stream);
  // Copy back key/value
  const auto d_position_current = position_a_.data().get() + segment.begin;
  const auto d_position_other = position_temp.data().get() + segment.begin;
  const auto d_ridx_current = ridx_a_.data().get() + segment.begin;
  const auto d_ridx_other = ridx_temp.data().get() + segment.begin;
  dh::LaunchN(device_idx_, segment.Size(), stream, [=] __device__(size_t idx) {
    d_position_current[idx] = d_position_other[idx];
    d_ridx_current[idx] = d_ridx_other[idx];
  });
}
};  // namespace tree
};  // namespace xgboost
