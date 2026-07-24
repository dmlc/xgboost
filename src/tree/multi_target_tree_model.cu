/**
 * Copyright 2026, XGBoost Contributors
 */
#include <xgboost/context.h>  // for Context
#include <xgboost/span.h>     // for Span

#include <cstddef>  // for size_t
#include <limits>   // for numeric_limits

#include "../common/cuda_context.cuh"    // for CUDAContext
#include "../common/device_helpers.cuh"  // for LaunchN, MemcpyBatchAsync

namespace xgboost::tree::cuda_impl {
void CopyBatch(Context const* ctx, common::Span<void*> dsts, common::Span<void const*> srcs,
               common::Span<std::size_t const> sizes) {
  std::size_t fail_idx{std::numeric_limits<std::size_t>::max()};
  auto status = dh::MemcpyBatchAsync<cudaMemcpyDeviceToDevice>(
      dsts.data(), srcs.data(), sizes.data(), dsts.size(), &fail_idx, ctx->CUDACtx()->Stream());
  if (status != cudaSuccess) {
    LOG(FATAL) << "CUDA batch copy failed at index " << fail_idx << ": "
               << cudaGetErrorString(status);
  }
}

void ApplyLearningRate(Context const* ctx, common::Span<float> weights, float eta) {
  dh::LaunchN(weights.size(), ctx->CUDACtx()->Stream(),
              [=] XGBOOST_DEVICE(std::size_t i) mutable { weights[i] *= eta; });
}
}  // namespace xgboost::tree::cuda_impl
