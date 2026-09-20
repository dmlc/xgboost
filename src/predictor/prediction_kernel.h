/**
 * Copyright 2026, XGBoost Contributors
 */
#ifndef XGBOOST_PREDICTOR_PREDICTION_KERNEL_H_
#define XGBOOST_PREDICTOR_PREDICTION_KERNEL_H_

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
}  // namespace predictor
}  // namespace xgboost

#endif  // XGBOOST_PREDICTOR_PREDICTION_KERNEL_H_
