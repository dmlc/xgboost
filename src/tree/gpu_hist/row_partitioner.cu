/**
 * Copyright 2017-2026, XGBoost contributors
 */
#include <thrust/sequence.h>  // for sequence

#include <vector>  // for vector

#include "../../common/cuda_context.cuh"    // for CUDAContext
#include "../../common/device_helpers.cuh"  // for CopyDeviceSpanToVector, ToSpan
#include "row_partitioner.cuh"

namespace xgboost::tree {
void RowPartitioner::InitRoot(bst_idx_t n_samples) {
  CHECK_LE(n_samples, std::numeric_limits<cuda_impl::RowIndexT>::max());
  ridx_segments_.clear();
  ridx_.resize(n_samples);
  tmp_.clear();
  n_nodes_ = 1;  // Root

  ridx_segments_.emplace_back(
      NodePositionInfo{Segment{0, static_cast<cuda_impl::RowIndexT>(n_samples)}});

  // Pre-allocate some host memory
  this->pinned_.GetSpan<std::int32_t>(1 << 11);
  this->pinned2_.GetSpan<std::int32_t>(1 << 13);
}

void RowPartitioner::Reset(Context const* ctx, bst_idx_t n_samples, bst_idx_t base_rowid) {
  if (n_samples != 0) {
    CHECK_LE(base_rowid, std::numeric_limits<RowIndexT>::max());
    CHECK_LE(n_samples - 1, std::numeric_limits<RowIndexT>::max() - base_rowid)
        << "Global row indices exceed the GPU partitioner's uint32 range.";
  }
  this->InitRoot(n_samples);
  thrust::sequence(ctx->CUDACtx()->CTP(), ridx_.data(), ridx_.data() + ridx_.size(), base_rowid);
}

RowPartitioner::~RowPartitioner() = default;

common::Span<const RowPartitioner::RowIndexT> RowPartitioner::GetRows(bst_node_t nidx) {
  auto segment = ridx_segments_.at(nidx).segment;
  return dh::ToSpan(ridx_).subspan(segment.begin, segment.Size());
}

common::Span<const RowPartitioner::RowIndexT> RowPartitioner::GetRows() const {
  return dh::ToSpan(ridx_);
}

std::vector<RowPartitioner::RowIndexT> RowPartitioner::GetRowsHost(bst_node_t nidx) {
  auto span = GetRows(nidx);
  std::vector<RowIndexT> rows(span.size());
  dh::CopyDeviceSpanToVector(&rows, span);
  return rows;
}
};  // namespace xgboost::tree
