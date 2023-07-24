/*!
 * Copyright 2017-2022 XGBoost contributors
 */
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sequence.h>

#include <vector>

#include "../../common/device_helpers.cuh"
#include "row_partitioner.cuh"

namespace xgboost {
namespace tree {

RowPartitioner::RowPartitioner(int device_idx, size_t num_rows)
    : device_idx_(device_idx), ridx_(num_rows), ridx_tmp_(num_rows) {
  dh::safe_cuda(cudaSetDevice(device_idx_));
  ridx_segments_.emplace_back(NodePositionInfo{Segment(0, num_rows)});
  thrust::sequence(thrust::device, ridx_.data(), ridx_.data() + ridx_.size());
  dh::safe_cuda(cudaStreamCreate(&stream_));
}

RowPartitioner::~RowPartitioner() {
  dh::safe_cuda(cudaSetDevice(device_idx_));
  dh::safe_cuda(cudaStreamDestroy(stream_));
}

common::Span<const RowPartitioner::RowIndexT> RowPartitioner::GetRows(bst_node_t nidx) {
  auto segment = ridx_segments_.at(nidx).segment;
  return dh::ToSpan(ridx_).subspan(segment.begin, segment.Size());
}

common::Span<const RowPartitioner::RowIndexT> RowPartitioner::GetRows() {
  return dh::ToSpan(ridx_);
}

std::vector<RowPartitioner::RowIndexT> RowPartitioner::GetRowsHost(bst_node_t nidx) {
  auto span = GetRows(nidx);
  std::vector<RowIndexT> rows(span.size());
  dh::CopyDeviceSpanToVector(&rows, span);
  return rows;
}

};  // namespace tree
};  // namespace xgboost
