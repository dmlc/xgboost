/**
 * Copyright 2017-2025, XGBoost contributors
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
  ridx_other_.resize(n_samples);
  tmp_.clear();
  n_nodes_ = 1;  // Root

  CHECK_LE(n_samples, std::numeric_limits<cuda_impl::RowIndexT>::max());
  ridx_segments_.emplace_back(
      NodePositionInfo{Segment{0, static_cast<cuda_impl::RowIndexT>(n_samples)}});

  thrust::sequence(ctx->CUDACtx()->CTP(), ridx_.data(), ridx_.data() + ridx_.size(), base_rowid);

  // Pre-allocate some host memory
  this->pinned_.GetSpan<std::int32_t>(1 << 11);
  this->pinned2_.GetSpan<std::int32_t>(1 << 13);
}

RowPartitioner::~RowPartitioner() = default;

common::Span<const RowPartitioner::RowIndexT> RowPartitioner::GetRows(bst_node_t nidx) {
  auto const& node = ridx_segments_.at(nidx);
  return this->Buffer(node.buffer).subspan(node.segment.begin, node.segment.Size());
}

void RowPartitioner::Consolidate(Context const* ctx) {
  for (auto& node : ridx_segments_) {
    if (node.IsLeaf() && node.buffer == 1 && node.segment.Size() != 0) {
      dh::safe_cuda(cudaMemcpyAsync(ridx_.data() + node.segment.begin,
                                    ridx_other_.data() + node.segment.begin,
                                    node.segment.Size() * sizeof(RowIndexT), cudaMemcpyDefault,
                                    ctx->CUDACtx()->Stream()));
      node.buffer = 0;
    }
  }
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
