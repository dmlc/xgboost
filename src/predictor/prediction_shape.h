/**
 * Copyright 2026, XGBoost Contributors
 */
#ifndef XGBOOST_PREDICTOR_PREDICTION_SHAPE_H_
#define XGBOOST_PREDICTOR_PREDICTION_SHAPE_H_

#include <vector>

#include "xgboost/base.h"

namespace xgboost::predictor {
/** @brief Shape of raw predictions, before the objective transform. */
inline void RawPredictionShape(bst_ulong rows, bst_ulong groups,
                               std::vector<bst_ulong>* out_shape) {
  if (out_shape) {
    *out_shape = {rows, groups};
  }
}

/** @brief SHAP outputs include a bias column and retain the output-group axis. */
inline void ContributionPredictionShape(bst_ulong rows, bst_ulong features, bst_ulong groups,
                                        bool interactions, std::vector<bst_ulong>* out_shape) {
  if (out_shape) {
    *out_shape = {rows, groups, features + 1};
    if (interactions) {
      out_shape->push_back(features + 1);
    }
  }
}

/** @brief Leaf dimensions are determined by the selected trees, even with no rows. */
inline void LeafPredictionShape(bst_ulong rows, bst_ulong trees, bst_ulong rounds, bst_ulong groups,
                                bst_ulong forest, bool strict_shape,
                                std::vector<bst_ulong>* out_shape) {
  if (out_shape) {
    if (strict_shape) {
      *out_shape = {rows, rounds, groups, forest};
    } else if (trees == 1) {
      *out_shape = {rows};
    } else {
      *out_shape = {rows, trees};
    }
  }
}
}  // namespace xgboost::predictor
#endif  // XGBOOST_PREDICTOR_PREDICTION_SHAPE_H_
