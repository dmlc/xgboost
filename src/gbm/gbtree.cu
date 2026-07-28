/**
 * Copyright 2021-2025, XGBoost Contributors
 */
#include <thrust/iterator/counting_iterator.h>  // for make_counting_iterator

#include "../common/cuda_context.cuh"
#include "../common/device_helpers.cuh"  // for MakeTransformIterator
#include "xgboost/base.h"                // for GradientPair
#include "xgboost/linalg.h"              // for Matrix

namespace xgboost::gbm {
void GPUCopyGradient(Context const *ctx, linalg::Matrix<GradientPair> const *in_gpair,
                     bst_target_t target_idx, linalg::Matrix<GradientPair> *out_gpair) {
  auto v_in = in_gpair->View(ctx->Device()).Slice(linalg::All(), target_idx);
  out_gpair->SetDevice(ctx->Device());
  out_gpair->Reshape(v_in.Size(), 1);
  auto d_out = out_gpair->View(ctx->Device());
  auto cuctx = ctx->CUDACtx();
  auto it = dh::MakeTransformIterator<GradientPair>(
      thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(std::size_t i) { return v_in(i); });
  thrust::copy(cuctx->CTP(), it, it + v_in.Size(), d_out.Values().data());
}

void GPUDartPredictInc(common::Span<float> out_predts, common::Span<float> predts, float tree_w,
                       size_t n_rows, bst_target_t n_targets, bst_target_t target_idx) {
  dh::LaunchN(n_rows, [=] XGBOOST_DEVICE(size_t ridx) {
    const size_t offset = ridx * n_targets + target_idx;
    out_predts[offset] += (predts[offset] * tree_w);
  });
}

void GPUDartInplacePredictInc(common::Span<float> out_predts, common::Span<float> predts,
                              float tree_w, size_t n_rows,
                              linalg::TensorView<float const, 1> base_score, bst_target_t n_targets,
                              bst_target_t target_idx) {
  CHECK_EQ(base_score.Size(), n_targets);
  dh::LaunchN(n_rows, [=] XGBOOST_DEVICE(size_t ridx) {
    const size_t offset = ridx * n_targets + target_idx;
    out_predts[offset] += (predts[offset] - base_score(target_idx)) * tree_w;
  });
}
}  // namespace xgboost::gbm
