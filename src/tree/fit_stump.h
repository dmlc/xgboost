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

template <typename T>
XGBOOST_DEVICE inline double CalcUnregulatedWeight(T sum_grad, T sum_hess) {
  return -sum_grad / std::max(sum_hess, static_cast<double>(kRtEps));
}

/**
 * @brief Fit a tree stump as an estimation of base_score.
 */
void FitStump(Context const* ctx, HostDeviceVector<GradientPair> const& gpair,
              bst_target_t n_targets, linalg::Vector<float>* out);
}  // namespace obj
}  // namespace xgboost
#endif  // XGBOOST_OBJECTIVE_INIT_ESTIMATION_H_
