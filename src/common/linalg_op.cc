/**
 * Copyright 2025, XGBoost Contributors
 */
#include "linalg_op.h"

#include "optional_weight.h"
#include "xgboost/context.h"  // for Context

namespace xgboost::linalg {
namespace cuda_impl {
void SmallHistogram(Context const* ctx, linalg::MatrixView<float const> indices,
                    common::OptionalWeights const& weights, linalg::VectorView<float> bins);
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
