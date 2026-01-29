/**
 * Copyright 2025, XGBoost Contributors
 */
#include "optional_weight.h"

#include <numeric>  // for accumulate

#include "xgboost/base.h"     // for bst_idx_t
#include "xgboost/context.h"  // for Context

#include "common.h"  // for AssertGPUSupport

namespace xgboost::common {
#if defined(XGBOOST_USE_CUDA)
namespace cuda_impl {
double SumOptionalWeights(Context const* ctx, OptionalWeights const& weights);
}
#endif

#if defined(XGBOOST_USE_SYCL)
namespace sycl_impl {
double SumOptionalWeights(Context const* ctx, OptionalWeights const& weights);
}
#endif

[[nodiscard]] double SumOptionalWeights(Context const* ctx, OptionalWeights const& weights,
                                        bst_idx_t n_samples) {
  if (weights.Empty()) {
    return n_samples * weights.dft;
  }
  if (ctx->IsCUDA()) {
#if defined(XGBOOST_USE_CUDA)
    return cuda_impl::SumOptionalWeights(ctx, weights);
#else
    common::AssertGPUSupport();
#endif
  }
  if (ctx->IsSycl()) {
#if defined(XGBOOST_USE_SYCL)
    return sycl_impl::SumOptionalWeights(ctx, weights);
#else
    common::AssertSYCLSupport();
#endif
  }
  auto sum_weight = std::accumulate(weights.Data(), weights.Data() + weights.Size(), 0.0);
  return sum_weight;
}
}  // namespace xgboost::common
