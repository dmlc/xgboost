/*!
 * Copyright 2017-2019 XGBoost contributors
 */
#include <thrust/sequence.h>
#include <vector>
#include "../../common/device_helpers.cuh"
#include "row_partitioner.cuh"

namespace xgboost {
namespace tree {

struct IndicateLeftTransform {
  bst_node_t left_nidx;
  explicit IndicateLeftTransform(bst_node_t left_nidx)
      : left_nidx(left_nidx) {}
  __host__ __device__ __forceinline__ int operator()(const bst_node_t& x) const {
    return x == left_nidx ? 1 : 0;
  }
};
/*
 * position: Position of rows belonged to current split node.
 */
void RowPartitioner::SortPosition(common::Span<bst_node_t> position,
                                  common::Span<bst_node_t> position_out,
                                  common::Span<RowIndexT> ridx,
                                  common::Span<RowIndexT> ridx_out,
                                  bst_node_t left_nidx,
                                  bst_node_t right_nidx,
                                  int64_t* d_left_count, cudaStream_t stream) {
  // radix sort over 1 bit, see:
  // https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html
  auto d_position_out = position_out.data();
  auto d_position_in = position.data();
  auto d_ridx_out = ridx_out.data();
  auto d_ridx_in = ridx.data();
  auto write_results = [=] __device__(size_t idx, int ex_scan_result) {
    // the ex_scan_result represents how many rows have been assigned to left node so far
    // during scan.
    int scatter_address;
    if (d_position_in[idx] == left_nidx) {
      scatter_address = ex_scan_result;
    } else {
      // current number of rows belong to right node + total number of rows belong to left
      // node
      scatter_address = (idx - ex_scan_result) + *d_left_count;
    }
    // copy the node id to output
    d_position_out[scatter_address] = d_position_in[idx];
    d_ridx_out[scatter_address] = d_ridx_in[idx];
  };  // NOLINT

  IndicateLeftTransform is_left(left_nidx);
  // an iterator that given a old position returns whether it belongs to left or right
  // node.
  cub::TransformInputIterator<bst_node_t, IndicateLeftTransform,
                              bst_node_t*>
      in_itr(d_position_in, is_left);
  dh::DiscardLambdaItr<decltype(write_results)> out_itr(write_results);
  size_t temp_storage_bytes = 0;
  // position is of the same size with current split node's row segment
  cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, in_itr, out_itr,
                                position.size(), stream);
  dh::caching_device_vector<uint8_t> temp_storage(temp_storage_bytes);
  cub::DeviceScan::ExclusiveSum(temp_storage.data().get(), temp_storage_bytes,
                                in_itr, out_itr, position.size(), stream);
}
RowPartitioner::RowPartitioner(int device_idx, size_t num_rows)
    : device_idx(device_idx) {
  dh::safe_cuda(cudaSetDevice(device_idx));
  ridx_a.resize(num_rows);
  ridx_b.resize(num_rows);
  position_a.resize(num_rows);
  position_b.resize(num_rows);
  ridx = dh::DoubleBuffer<RowIndexT>{&ridx_a, &ridx_b};
  position = dh::DoubleBuffer<bst_node_t>{&position_a, &position_b};
  ridx_segments.emplace_back(Segment(0, num_rows));

  thrust::sequence(
      thrust::device_pointer_cast(ridx.CurrentSpan().data()),
      thrust::device_pointer_cast(ridx.CurrentSpan().data() + ridx.Size()));
  thrust::fill(
      thrust::device_pointer_cast(position.Current()),
      thrust::device_pointer_cast(position.Current() + position.Size()), 0);
  left_counts.resize(256);
  thrust::fill(left_counts.begin(), left_counts.end(), 0);
  streams.resize(2);
  for (auto& stream : streams) {
    dh::safe_cuda(cudaStreamCreate(&stream));
  }
}
RowPartitioner::~RowPartitioner() {
  dh::safe_cuda(cudaSetDevice(device_idx));
  for (auto& stream : streams) {
    dh::safe_cuda(cudaStreamDestroy(stream));
  }
}

common::Span<const RowPartitioner::RowIndexT> RowPartitioner::GetRows(
    bst_node_t nidx) {
  auto segment = ridx_segments.at(nidx);
  // Return empty span here as a valid result
  // Will error if we try to construct a span from a pointer with size 0
  if (segment.Size() == 0) {
    return common::Span<const RowPartitioner::RowIndexT>();
  }
  return ridx.CurrentSpan().subspan(segment.begin, segment.Size());
}

common::Span<const RowPartitioner::RowIndexT> RowPartitioner::GetRows() {
  return ridx.CurrentSpan();
}

common::Span<const bst_node_t> RowPartitioner::GetPosition() {
  return position.CurrentSpan();
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
      common::Span<bst_node_t>(position.Current() + segment.begin,
                                  segment.Size()),
      // position_out
      common::Span<bst_node_t>(position.other() + segment.begin,
                                  segment.Size()),
      // row index in
      common::Span<RowIndexT>(ridx.Current() + segment.begin, segment.Size()),
      // row index out
      common::Span<RowIndexT>(ridx.other() + segment.begin, segment.Size()),
      left_nidx, right_nidx, d_left_count, stream);
  // Copy back key/value
  const auto d_position_current = position.Current() + segment.begin;
  const auto d_position_other = position.other() + segment.begin;
  const auto d_ridx_current = ridx.Current() + segment.begin;
  const auto d_ridx_other = ridx.other() + segment.begin;
  dh::LaunchN(device_idx, segment.Size(), stream, [=] __device__(size_t idx) {
    d_position_current[idx] = d_position_other[idx];
    d_ridx_current[idx] = d_ridx_other[idx];
  });
}
};  // namespace tree
};  // namespace xgboost
