/**
 * Copyright 2025, XGBoost Contributors
 */
#include "linalg_op.h"

#include <cstddef>  // for size_t

#include "optional_weight.h"  // for OptionalWeights
#include "xgboost/context.h"  // for Context

#if !defined(XGBOOST_USE_CUDA) && !defined(XGBOOST_USE_SYCL)
#include "common.h"  // for AssertGPUSupport
#endif

namespace xgboost::sycl::linalg {
void SmallHistogram(Context const* ctx, xgboost::linalg::MatrixView<float const> indices,
                    common::OptionalWeights const& weights,
                    xgboost::linalg::VectorView<float> bins);
#if !defined(XGBOOST_USE_SYCL)
void SmallHistogram(Context const*, xgboost::linalg::MatrixView<float const>,
                    common::OptionalWeights const&,
                    xgboost::linalg::VectorView<float>) {
  common::AssertSYCLSupport();
}
#endif
}  // namespace xgboost::sycl::linalg

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
  if (ctx->IsCUDA()) {
    cuda_impl::SmallHistogram(ctx, indices, weights, bins);
  } else if (ctx->IsSycl()) {
    sycl::linalg::SmallHistogram(ctx, indices, weights, bins);
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      auto y = indices(i);
      auto w = weights[i];
      bins(static_cast<std::size_t>(y)) += w;
    }
  }
}
}  // namespace xgboost::linalg
