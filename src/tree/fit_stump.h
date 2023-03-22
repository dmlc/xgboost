/**
 * Copyright 2022 by XGBoost Contributors
 *
 * \brief Utilities for estimating initial score.
 */

#ifndef XGBOOST_TREE_FIT_STUMP_H_
#define XGBOOST_TREE_FIT_STUMP_H_

#if !defined(NOMINMAX) && defined(_WIN32)
#define NOMINMAX
#endif  // !defined(NOMINMAX)

#include <algorithm>  // std::max

#include "../common/common.h"            // AssertGPUSupport
#include "xgboost/base.h"                // GradientPair
#include "xgboost/context.h"             // Context
#include "xgboost/data.h"                // MetaInfo
#include "xgboost/host_device_vector.h"  // HostDeviceVector
#include "xgboost/linalg.h"              // TensorView

namespace xgboost {
namespace tree {

template <typename T>
XGBOOST_DEVICE inline double CalcUnregularizedWeight(T sum_grad, T sum_hess) {
  return -sum_grad / std::max(sum_hess, static_cast<double>(kRtEps));
}

/**
 * @brief Fit a tree stump as an estimation of base_score.
 */
void FitStump(Context const* ctx, MetaInfo const& info, HostDeviceVector<GradientPair> const& gpair,
              bst_target_t n_targets, linalg::Vector<float>* out);
}  // namespace tree
}  // namespace xgboost
#endif  // XGBOOST_TREE_FIT_STUMP_H_
