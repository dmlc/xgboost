/*!
 * Copyright 2017-2021 XGBoost contributors
 */
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sequence.h>
#include <vector>
#include "../../common/device_helpers.cuh"
#include "row_partitioner.cuh"

namespace xgboost {
namespace tree {
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

// Implement partitioning via single scan operation using transform output to
// write the result
void RowPartitioner::SortPosition(common::Span<bst_node_t> position,
                                  common::Span<bst_node_t> position_out,
                                  common::Span<RowIndexT> ridx,
                                  common::Span<RowIndexT> ridx_out,
                                  bst_node_t left_nidx, bst_node_t,
                                  int64_t* d_left_count, cudaStream_t stream) {
  WriteResultsFunctor write_results{left_nidx, position, position_out,
                                    ridx,      ridx_out, d_left_count};
  auto discard_write_iterator =
      thrust::make_transform_output_iterator(dh::TypedDiscard<IndexFlagTuple>(), write_results);
  auto counting = thrust::make_counting_iterator(0llu);
  auto input_iterator = dh::MakeTransformIterator<IndexFlagTuple>(
      counting, [=] __device__(size_t idx) {
        return IndexFlagTuple{idx, static_cast<size_t>(position[idx] == left_nidx)};
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
  dh::LaunchN(ridx.size(), [=] __device__(size_t idx) {
    ridx[idx] = idx;
    position[idx] = 0;
  });
}

RowPartitioner::RowPartitioner(int device_idx, size_t num_rows)
    : device_idx_(device_idx), ridx_a_(num_rows), position_a_(num_rows),
      ridx_b_(num_rows), position_b_(num_rows) {
  dh::safe_cuda(cudaSetDevice(device_idx_));
  ridx_ = dh::DoubleBuffer<RowIndexT>{&ridx_a_, &ridx_b_};
  position_ = dh::DoubleBuffer<bst_node_t>{&position_a_, &position_b_};
  ridx_segments_.emplace_back(Segment(0, num_rows));

  Reset(device_idx, ridx_.CurrentSpan(), position_.CurrentSpan());
  left_counts_.resize(256);
  thrust::fill(left_counts_.begin(), left_counts_.end(), 0);
  streams_.resize(2);
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
    return {};
  }
  return ridx_.CurrentSpan().subspan(segment.begin, segment.Size());
}

common::Span<const RowPartitioner::RowIndexT> RowPartitioner::GetRows() {
  return ridx_.CurrentSpan();
}

common::Span<const bst_node_t> RowPartitioner::GetPosition() {
  return position_.CurrentSpan();
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
  SortPosition(
      // position_in
      common::Span<bst_node_t>(position_.Current() + segment.begin,
                               segment.Size()),
      // position_out
      common::Span<bst_node_t>(position_.Other() + segment.begin,
                               segment.Size()),
      // row index in
      common::Span<RowIndexT>(ridx_.Current() + segment.begin, segment.Size()),
      // row index out
      common::Span<RowIndexT>(ridx_.Other() + segment.begin, segment.Size()),
      left_nidx, right_nidx, d_left_count, stream);
  // Copy back key/value
  const auto d_position_current = position_.Current() + segment.begin;
  const auto d_position_other = position_.Other() + segment.begin;
  const auto d_ridx_current = ridx_.Current() + segment.begin;
  const auto d_ridx_other = ridx_.Other() + segment.begin;
  dh::LaunchN(segment.Size(), stream, [=] __device__(size_t idx) {
    d_position_current[idx] = d_position_other[idx];
    d_ridx_current[idx] = d_ridx_other[idx];
  });
}
};  // namespace tree
};  // namespace xgboost
