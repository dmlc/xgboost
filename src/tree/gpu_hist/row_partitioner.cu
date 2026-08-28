/**
 * Copyright 2017-2025, XGBoost contributors
 */
#include <thrust/sequence.h>  // for sequence

#include <vector>  // for vector

#include "../../common/cuda_context.cuh"    // for CUDAContext
#include "../../common/device_helpers.cuh"  // for CopyDeviceSpanToVector, ToSpan
#include "row_partitioner.cuh"

namespace xgboost::tree {
template <typename OpDataT>
__global__ void BuildBatchInfoKernel(common::Span<NodePositionInfo const> d_ridx_segments,
                                     common::Span<bst_node_t const> nidx,
                                     common::Span<OpDataT const> op_data,
                                     common::Span<PerNodeData<OpDataT>> d_batch_info) {
  for (auto i : dh::GridStrideRange<std::size_t>(0, nidx.size())) {
    d_batch_info[i] = {d_ridx_segments[nidx[i]].segment, op_data[i]};
  }
}

__global__ void UpdateSegmentsKernel(common::Span<NodePositionInfo> d_ridx_segments,
                                     common::Span<bst_node_t const> nidx,
                                     common::Span<bst_node_t const> left_nidx,
                                     common::Span<bst_node_t const> right_nidx,
                                     common::Span<cuda_impl::RowIndexT const> counts) {
  for (auto i : dh::GridStrideRange<std::size_t>(0, nidx.size())) {
    auto segment = d_ridx_segments[nidx[i]].segment;
    auto left_count = counts[i];
    d_ridx_segments[nidx[i]] = {segment, left_nidx[i], right_nidx[i]};
    d_ridx_segments[left_nidx[i]] = {Segment{segment.begin, segment.begin + left_count}};
    d_ridx_segments[right_nidx[i]] = {Segment{segment.begin + left_count, segment.end}};
  }
}

void RowPartitioner::Reset(Context const* ctx, bst_idx_t n_samples, bst_idx_t base_rowid) {
  ridx_segments_.clear();
  ridx_.resize(n_samples);
  counts_.clear();
  tmp_.clear();
  n_nodes_ = 1;  // Root

  CHECK_LE(n_samples, std::numeric_limits<cuda_impl::RowIndexT>::max());
  ridx_segments_.emplace_back(
      NodePositionInfo{Segment{0, static_cast<cuda_impl::RowIndexT>(n_samples)}});
  d_ridx_segments_.resize(1);
  dh::safe_cuda(cudaMemcpyAsync(d_ridx_segments_.data(), ridx_segments_.data(),
                                sizeof(NodePositionInfo), cudaMemcpyDefault,
                                ctx->CUDACtx()->Stream()));

  thrust::sequence(ctx->CUDACtx()->CTP(), ridx_.data(), ridx_.data() + ridx_.size(), base_rowid);

  // Pre-allocate some host memory
  this->pinned_.GetSpan<std::int32_t>(1 << 11);
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
