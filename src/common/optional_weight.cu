/**
 * Copyright 2025, XGBoost Contributors
 */
#include <cuda/std/functional>  // for plus

#include <cstddef>              // for size_t

#include "cuda_context.cuh"
#include "device_helpers.cuh"
#include "optional_weight.h"
#include "xgboost/context.h"  // for Context

namespace xgboost::common::cuda_impl {
double SumOptionalWeights(Context const* ctx, OptionalWeights const& weights) {
  auto w_it = dh::MakeIndexTransformIter([=] XGBOOST_DEVICE(std::size_t i) { return weights[i]; });
  return dh::Reduce(ctx->CUDACtx()->CTP(), w_it, w_it + weights.Size(), 0.0, cuda::std::plus{});
}
}  // namespace xgboost::common::cuda_impl
