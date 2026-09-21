/**
 * Copyright 2026, XGBoost Contributors
 */
#ifndef XGBOOST_PREDICTOR_PREDICTION_KERNEL_H_
#define XGBOOST_PREDICTOR_PREDICTION_KERNEL_H_

#include <vector>  // for vector

#include "xgboost/base.h"  // for bst_tree_t

namespace xgboost {
class Context;
class DMatrix;
template <typename T>
class HostDeviceVector;
namespace gbm {
struct GBTreeModel;
}  // namespace gbm

namespace predictor {
/**
 * \brief Write row-major leaf indices for the first tree_end trees.
 *
 * Zero or a limit larger than the model selects all trees. The output is resized and
 * overwritten; no prediction cache or base margin is used.
 */
struct PredictLeafKernel {
  using Signature = void(Context const*, DMatrix*, HostDeviceVector<float>*,
                         gbm::GBTreeModel const&, bst_tree_t tree_end);
};
/**
 * \brief Write exact SHAP contributions using explicit tree weights.
 *
 * Zero tree_end selects all trees; null tree_weights gives all trees unit weight.
 * CPU supports conditional SHAP; CUDA requires condition and condition_feature to be zero.
 */
struct PredictContributionKernel {
  using Signature = void(Context const*, DMatrix*, HostDeviceVector<float>*,
                         gbm::GBTreeModel const&, bst_tree_t tree_end,
                         std::vector<float> const* tree_weights, int condition,
                         unsigned condition_feature);
};
/**
 * \brief Write approximate contributions. CUDA reports this operation as unsupported.
 *
 * Zero tree_end selects all trees; null tree_weights gives all trees unit weight.
 */
struct PredictApproxContributionKernel {
  using Signature = void(Context const*, DMatrix*, HostDeviceVector<float>*,
                         gbm::GBTreeModel const&, bst_tree_t tree_end,
                         std::vector<float> const* tree_weights);
};
/**
 * \brief Write SHAP interaction contributions using explicit tree weights.
 *
 * Zero tree_end selects all trees. CPU supports exact and approximate interactions;
 * CUDA supports exact interactions only. Null tree_weights gives all trees unit weight.
 */
struct PredictInteractionContributionsKernel {
  using Signature = void(Context const*, DMatrix*, HostDeviceVector<float>*,
                         gbm::GBTreeModel const&, bst_tree_t tree_end,
                         std::vector<float> const* tree_weights, bool approximate);
};
}  // namespace predictor
}  // namespace xgboost

#endif  // XGBOOST_PREDICTOR_PREDICTION_KERNEL_H_
