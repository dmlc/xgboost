/**
 * Copyright 2022 by XGBoost Contributors
 *
 * \brief Utilities for estimating initial score.
 */

#ifndef XGBOOST_OBJECTIVE_INIT_ESTIMATION_H_
#define XGBOOST_OBJECTIVE_INIT_ESTIMATION_H_

#include "../common/common.h"            // AssertGPUSupport
#include "xgboost/data.h"                // MetaInfo
#include "xgboost/generic_parameters.h"  // Context

namespace xgboost {
namespace obj {
namespace cpu_impl {
double WeightedMean(Context const* ctx, MetaInfo const& info);
}  // namespace cpu_impl

namespace cuda_impl {
double WeightedMean(Context const* ctx, MetaInfo const& info);
#if !defined(XGBOOST_USE_CUDA)
inline double WeightedMean(Context const*, MetaInfo const&) {
  common::AssertGPUSupport();
  return 0.0;
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace cuda_impl

/**
 * \brief Weighted mean for distributed env. Not a general implementation since we have
 *        2-dim label with 1-dim weight.
 */
inline double WeightedMean(Context const* ctx, MetaInfo const& info) {
  return ctx->IsCPU() ? cpu_impl::WeightedMean(ctx, info) : cuda_impl::WeightedMean(ctx, info);
}
}  // namespace obj
}  // namespace xgboost
#endif  // XGBOOST_OBJECTIVE_INIT_ESTIMATION_H_
