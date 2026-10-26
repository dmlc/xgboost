/**
 * Copyright 2026, XGBoost Contributors
 */
#ifndef XGBOOST_PREDICTOR_PREDICTION_SHAPE_H_
#define XGBOOST_PREDICTOR_PREDICTION_SHAPE_H_

#include <vector>

#include "xgboost/base.h"

namespace xgboost::predictor {
/**
 * @brief SHAP outputs include a bias column and retain the output-group axis.
 * @return `(rows, groups, features + 1)`, with a further `features + 1` axis for
 *         interactions. Trailing dimensions are preserved for zero rows.
 */
inline std::vector<bst_ulong> ContributionPredictionShape(bst_ulong rows, bst_ulong features,
                                                          bst_ulong groups, bool interactions) {
  std::vector<bst_ulong> shape{rows, groups, features + 1};
  if (interactions) {
    shape.push_back(features + 1);
  }
  return shape;
}

/**
 * @brief Leaf dimensions are determined by the selected trees, even with no rows.
 * @return `(rows, rounds, groups, forest)` with strict_shape; otherwise `(rows, trees)`,
 *         or `(rows,)` for one tree. Trailing dimensions are preserved for zero rows.
 */
inline std::vector<bst_ulong> LeafPredictionShape(bst_ulong rows, bst_ulong trees, bst_ulong rounds,
                                                  bst_ulong groups, bst_ulong forest,
                                                  bool strict_shape) {
  if (strict_shape) {
    return {rows, rounds, groups, forest};
  }
  if (trees == 1) {
    return {rows};
  }
  return {rows, trees};
}
}  // namespace xgboost::predictor
#endif  // XGBOOST_PREDICTOR_PREDICTION_SHAPE_H_
