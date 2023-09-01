/**
 * Copyright 2021-2023, XGBoost Contributors
 */
#include <thrust/iterator/counting_iterator.h>  // for make_counting_iterator

#include "../common/cuda_context.cuh"
#include "../common/device_helpers.cuh"  // for MakeTransformIterator
#include "xgboost/base.h"                // for GradientPair
#include "xgboost/linalg.h"              // for Matrix

namespace xgboost::gbm {
void GPUCopyGradient(Context const *ctx, linalg::Matrix<GradientPair> const *in_gpair,
                     bst_group_t group_id, linalg::Matrix<GradientPair> *out_gpair) {
  auto v_in = in_gpair->View(ctx->Device()).Slice(linalg::All(), group_id);
  out_gpair->SetDevice(ctx->Device());
  out_gpair->Reshape(v_in.Size(), 1);
  auto d_out = out_gpair->View(ctx->Device());
  auto cuctx = ctx->CUDACtx();
  auto it = dh::MakeTransformIterator<GradientPair>(
      thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(std::size_t i) { return v_in(i); });
  thrust::copy(cuctx->CTP(), it, it + v_in.Size(), d_out.Values().data());
}

void GPUDartPredictInc(common::Span<float> out_predts,
                       common::Span<float> predts, float tree_w, size_t n_rows,
                       bst_group_t n_groups, bst_group_t group) {
  dh::LaunchN(n_rows, [=] XGBOOST_DEVICE(size_t ridx) {
    const size_t offset = ridx * n_groups + group;
    out_predts[offset] += (predts[offset] * tree_w);
  });
}

void GPUDartInplacePredictInc(common::Span<float> out_predts, common::Span<float> predts,
                              float tree_w, size_t n_rows,
                              linalg::TensorView<float const, 1> base_score, bst_group_t n_groups,
                              bst_group_t group) {
  CHECK_EQ(base_score.Size(), 1);
  dh::LaunchN(n_rows, [=] XGBOOST_DEVICE(size_t ridx) {
    const size_t offset = ridx * n_groups + group;
    out_predts[offset] += (predts[offset] - base_score(0)) * tree_w;
  });
}
}  // namespace xgboost::gbm
