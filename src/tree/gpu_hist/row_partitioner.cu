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
void Reset(int device_idx, common::Span<RowPartitioner::RowIndexT> ridx,
           common::Span<bst_node_t> position) {
  CHECK_EQ(ridx.size(), position.size());
  dh::LaunchN(ridx.size(), [=] __device__(size_t idx) {
    ridx[idx] = idx;
    position[idx] = 0;
  });
}

RowPartitioner::RowPartitioner(int device_idx, size_t num_rows)
    : device_idx_(device_idx), ridx_(num_rows),ridx_tmp_(num_rows),position_(num_rows) {
  dh::safe_cuda(cudaSetDevice(device_idx_));
  ridx_segments_.emplace_back(Segment(0, num_rows));

  Reset(device_idx, dh::ToSpan(ridx_), dh::ToSpan(position_));
}

common::Span<const RowPartitioner::RowIndexT> RowPartitioner::GetRows(
    bst_node_t nidx) {
  auto segment = ridx_segments_.at(nidx);
  // Return empty span here as a valid result
  // Will error if we try to construct a span from a pointer with size 0
  if (segment.Size() == 0) {
    return {};
  }
  return dh::ToSpan(ridx_).subspan(segment.begin, segment.Size());
}

common::Span<const RowPartitioner::RowIndexT> RowPartitioner::GetRows() {
  return dh::ToSpan(ridx_);
}

common::Span<const bst_node_t> RowPartitioner::GetPosition() {
  return dh::ToSpan(position_);
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

};  // namespace tree
};  // namespace xgboost
