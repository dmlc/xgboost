/**
 * Copyright 2017-2024, XGBoost contributors
 */
#include <thrust/sequence.h>  // for sequence

#include <vector>  // for vector

#include "../../common/cuda_context.cuh"    // for CUDAContext
#include "../../common/device_helpers.cuh"  // for CopyDeviceSpanToVector, ToSpan
#include "row_partitioner.cuh"

namespace xgboost::tree {
void RowPartitioner::Reset(Context const* ctx, bst_idx_t n_samples, bst_idx_t base_rowid) {
  ridx_segments_.clear();
  ridx_.resize(n_samples);
  ridx_tmp_.resize(n_samples);
  tmp_.clear();

  CHECK_LE(n_samples, std::numeric_limits<cuda_impl::RowIndexT>::max());
  ridx_segments_.emplace_back(
      NodePositionInfo{Segment{0, static_cast<cuda_impl::RowIndexT>(n_samples)}});

  thrust::sequence(ctx->CUDACtx()->CTP(), ridx_.data(), ridx_.data() + ridx_.size(), base_rowid);
}

RowPartitioner::~RowPartitioner() = default;

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
};  // namespace xgboost::tree
