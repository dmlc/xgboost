/**
 * Copyright 2026, XGBoost Contributors
 */
#include <xgboost/context.h>  // for Context
#include <xgboost/span.h>     // for Span

#include <cstddef>  // for size_t
#include <limits>   // for numeric_limits

#include "../common/cuda_context.cuh"         // for CUDAContext
#include "../common/device_helpers.cuh"       // for MemcpyBatchAsync
#include "xgboost/multi_target_tree_model.h"  // for CatWordT

namespace xgboost::tree::cuda_impl {
template <typename T>
void CopyBatch(Context const* ctx, common::Span<T*> dsts, common::Span<T const*> srcs,
               common::Span<std::size_t const> sizes) {
  std::size_t fail_idx{std::numeric_limits<std::size_t>::max()};
  dh::safe_cuda(dh::MemcpyBatchAsync<cudaMemcpyDeviceToDevice>(
      dsts.data(), srcs.data(), sizes.data(), dsts.size(), &fail_idx, ctx->CUDACtx()->Stream()));
}

template void CopyBatch(Context const* ctx, common::Span<float*> dsts,
                        common::Span<float const*> srcs, common::Span<std::size_t const> sizes);
template void CopyBatch(Context const* ctx, common::Span<CatWordT*> dsts,
                        common::Span<CatWordT const*> srcs, common::Span<std::size_t const> sizes);

}  // namespace xgboost::tree::cuda_impl
