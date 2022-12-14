/**
 * Copyright 2022 by XGBoost Contributors
 *
 * \brief Utilities for estimating initial score.
 */

#ifndef XGBOOST_OBJECTIVE_INIT_ESTIMATION_H_
#define XGBOOST_OBJECTIVE_INIT_ESTIMATION_H_

#include "../common/common.h"            // AssertGPUSupport
#include "xgboost/base.h"                // GradientPair
#include "xgboost/context.h"             // Context
#include "xgboost/host_device_vector.h"  // HostDeviceVector
#include "xgboost/linalg.h"              // TensorView

namespace xgboost {
namespace obj {
namespace cuda_impl {
double FitStump(Context const* ctx, HostDeviceVector<GradientPair> const& gpair);
#if !defined(XGBOOST_USE_CUDA)
inline double FitStump(Context const*, HostDeviceVector<GradientPair> const&) {
  common::AssertGPUSupport();
  return 0.0;
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace cuda_impl
/**
 * @brief Fit a tree stump as an estimation of base_score.
 */
double FitStump(Context const* ctx, HostDeviceVector<GradientPair> const& gpair);
}  // namespace obj
}  // namespace xgboost
#endif  // XGBOOST_OBJECTIVE_INIT_ESTIMATION_H_
