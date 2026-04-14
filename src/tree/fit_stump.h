/**
 * Copyright 2022-2026, XGBoost Contributors
 *
 * @brief Utilities for estimating initial score.
 */

#ifndef XGBOOST_TREE_FIT_STUMP_H_
#define XGBOOST_TREE_FIT_STUMP_H_

#include <algorithm>  // std::max

#include "xgboost/base.h"     // GradientPair
#include "xgboost/context.h"  // Context
#include "xgboost/data.h"     // MetaInfo
#include "xgboost/linalg.h"   // TensorView

namespace xgboost::tree {
namespace cpu_impl {
/**
 * @brief Sum gradients for each target.
 */
void SumGradients(Context const* ctx, linalg::MatrixView<GradientPair const> gpair,
                  linalg::VectorView<GradientPairPrecise> out);
}  // namespace cpu_impl

template <typename T>
XGBOOST_DEVICE inline double CalcUnregularizedWeight(T sum_grad, T sum_hess) {
  return -sum_grad / std::max(sum_hess, static_cast<double>(kRtEps));
}

/**
 * @brief Fit a tree stump as an estimation of base_score.
 */
void FitStump(Context const* ctx, MetaInfo const& info, linalg::Matrix<GradientPair> const& gpair,
              bst_target_t n_targets, linalg::Vector<float>* out);
}  // namespace xgboost::tree
#endif  // XGBOOST_TREE_FIT_STUMP_H_
