/**
 * Copyright 2017-2026, XGBoost contributors
 */
#include <thrust/sequence.h>   // for sequence
#include <thrust/transform.h>  // for transform

#include <cuda/functional>  // for proclaim_copyable_arguments
#include <vector>           // for vector

#include "../../common/cuda_context.cuh"    // for CUDAContext
#include "../../common/device_helpers.cuh"  // for CopyDeviceSpanToVector, ToSpan
#include "row_partitioner.cuh"

namespace xgboost::tree {
void RowPartitioner::Reset(Context const* ctx, bst_idx_t n_samples, bst_idx_t base_rowid) {
  ridx_segments_.clear();
  ridx_.resize(n_samples);
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

void RowPartitioner::Reset(Context const* ctx, bst_idx_t n_total_samples,
                           common::Span<bst_idx_t const> ridx) {
  ridx_segments_.clear();
  ridx_.resize(ridx.size());
  tmp_.clear();
  n_nodes_ = 1;  // Root

  // Unlike the base row index overload, `n_total_samples` is not the size of this
  // partitioner. It is the size of the index space the rows are drawn from: an out-of-range
  // row is an out-of-bounds access in every consumer, so the values are checked below.
  CHECK_LE(n_total_samples, std::numeric_limits<cuda_impl::RowIndexT>::max());
  CHECK_LE(ridx.size(), n_total_samples);
  ridx_segments_.emplace_back(
      NodePositionInfo{Segment{0, static_cast<cuda_impl::RowIndexT>(ridx.size())}});

  thrust::transform(
      ctx->CUDACtx()->CTP(), dh::tcbegin(ridx), dh::tcend(ridx), ridx_.data(),
      cuda::proclaim_copyable_arguments([=] XGBOOST_DEVICE(bst_idx_t r) -> cuda_impl::RowIndexT {
        return static_cast<cuda_impl::RowIndexT>(r);
      }));

  // Pre-allocate some host memory
  this->pinned_.GetSpan<std::int32_t>(1 << 11);
  this->pinned2_.GetSpan<std::int32_t>(1 << 13);
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
