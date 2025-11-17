/**
 * Copyright 2025, XGBoost contributors
 */
#include <cstddef>  // for size_t
#include <vector>   // for vector

#include "../../common/linalg_op.cuh"  // for tbegin
#include "../updater_gpu_common.cuh"   // for GPUTrainingParam
#include "leaf_sum.cuh"
#include "quantiser.cuh"        // for GradientQuantiser
#include "row_partitioner.cuh"  // for RowIndexT, LeafInfo
#include "xgboost/base.h"       // for GradientPairInt64
#include "xgboost/context.h"    // for Context
#include "xgboost/linalg.h"     // for MatrixView
#include "xgboost/span.h"       // for Span

namespace xgboost::tree::cuda_impl {
void LeafGradSum(Context const* ctx, std::vector<LeafInfo> const& h_leaves,
                 common::Span<GradientQuantiser const> roundings,
                 common::Span<RowIndexT const> sorted_ridx,
                 linalg::MatrixView<GradientPair const> grad,
                 linalg::MatrixView<GradientPairInt64> out_sum) {
  CHECK_EQ(h_leaves.size(), out_sum.Shape(0));

  dh::device_vector<LeafInfo> leaves(h_leaves);
  auto d_leaves = dh::ToSpan(leaves);

  std::vector<RowIndexT> h_indptr{0};
  for (auto const& node : h_leaves) {
    h_indptr.push_back(node.node.segment.Size());
  }
  // leaves form a complete partition
  dh::device_vector<RowIndexT> indptr{h_indptr};
  thrust::inclusive_scan(ctx->CUDACtx()->CTP(), indptr.cbegin(), indptr.cend(), indptr.begin());
  CHECK_EQ(roundings.size(), grad.Shape(1));
  CHECK_EQ(roundings.size(), out_sum.Shape(1));
  CHECK_EQ(out_sum.Shape(0), indptr.size() - 1);
  CHECK_EQ(indptr.size(), h_leaves.size() + 1);
  auto d_indptr = dh::ToSpan(indptr);

  for (bst_target_t t = 0, n_targets = grad.Shape(1); t < n_targets; ++t) {
    auto out_t = out_sum.Slice(linalg::All(), t);  // len == n_leaves
    auto it = dh::MakeIndexTransformIter([=] XGBOOST_DEVICE(std::size_t i) {
      auto nidx_in_set = dh::SegmentId(d_indptr, i);
      // Index within segment
      auto k = i - d_indptr[nidx_in_set];
      // Global index (within a batch).
      auto j = d_leaves[nidx_in_set].node.segment.begin + k;
      // gradient
      auto g = grad(sorted_ridx[j], t);
      return roundings[t].ToFixedPoint(g);
    });
    std::size_t n_bytes = 0;
    dh::safe_cuda(cub::DeviceSegmentedReduce::Sum(nullptr, n_bytes, it, linalg::tbegin(out_t),
                                                  h_leaves.size(), indptr.data(), indptr.data() + 1,
                                                  ctx->CUDACtx()->Stream()));
    dh::TemporaryArray<char> alloc(n_bytes);
    dh::safe_cuda(cub::DeviceSegmentedReduce::Sum(
        alloc.data().get(), n_bytes, it, linalg::tbegin(out_t), h_leaves.size(), indptr.data(),
        indptr.data() + 1, ctx->CUDACtx()->Stream()));
  }
}

void LeafWeight(Context const* ctx, GPUTrainingParam const& param,
                common::Span<GradientQuantiser const> roundings,
                linalg::MatrixView<GradientPairInt64 const> grad_sum,
                linalg::MatrixView<float> out_weights) {
  CHECK(grad_sum.Contiguous());
  auto s_grad_sum = grad_sum.Values();
  dh::LaunchN(grad_sum.Size(), ctx->CUDACtx()->Stream(), [=] XGBOOST_DEVICE(std::size_t i) mutable {
    auto [nidx_in_set, t] = linalg::UnravelIndex(i, grad_sum.Shape());
    auto g = roundings[t].ToFloatingPoint(grad_sum(nidx_in_set, t));
    out_weights(nidx_in_set, t) = CalcWeight(param, g.GetGrad(), g.GetHess());
  });
}
}  // namespace xgboost::tree::cuda_impl
