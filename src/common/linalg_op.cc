/**
 * Copyright 2025, XGBoost Contributors
 */
#include "linalg_op.h"

#include <cstddef>  // for size_t

#include "optional_weight.h"  // for OptionalWeights
#include "xgboost/context.h"  // for Context

#if !defined(XGBOOST_USE_CUDA)
#include "common.h"  // for AssertGPUSupport
#endif

namespace xgboost::linalg {
namespace cuda_impl {
void SmallHistogram(Context const* ctx, linalg::MatrixView<float const> indices,
                    common::OptionalWeights const& weights, linalg::VectorView<float> bins);
#if !defined(XGBOOST_USE_CUDA)
void SmallHistogram(Context const*, linalg::MatrixView<float const>, common::OptionalWeights const&,
                    linalg::VectorView<float>) {
  common::AssertGPUSupport();
}
#endif
}  // namespace cuda_impl

void SmallHistogram(Context const* ctx, linalg::MatrixView<float const> indices,
                    common::OptionalWeights const& weights, linalg::VectorView<float> bins) {
  auto n = indices.Size();
  if (!ctx->IsCUDA()) {
    for (std::size_t i = 0; i < n; ++i) {
      auto y = indices(i);
      auto w = weights[i];
      bins(static_cast<std::size_t>(y)) += w;
    }
  } else {
    cuda_impl::SmallHistogram(ctx, indices, weights, bins);
  }
}
}  // namespace xgboost::linalg
