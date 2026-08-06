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

void GPUScalePrediction(common::Span<float> predictions, float scale) {
  dh::LaunchN(predictions.size(),
              [=] XGBOOST_DEVICE(size_t i) { predictions[i] *= scale; });
}
}  // namespace xgboost::gbm
