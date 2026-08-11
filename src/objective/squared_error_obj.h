/**
 * Copyright 2026, XGBoost Contributors
 * \file squared_error_obj.h
 * \brief Shared declarations for the squared-error objective.
 */
#ifndef XGBOOST_OBJECTIVE_SQUARED_ERROR_OBJ_H_
#define XGBOOST_OBJECTIVE_SQUARED_ERROR_OBJ_H_

#include "elementwise_objective.h"  // for elementwise kernels
#include "xgboost/base.h"           // for GradientPair

namespace xgboost::obj {
struct SquaredErrorGradient {
  float scale_pos_weight;

  XGBOOST_DEVICE GradientPair operator()(float predt, float label, float weight) const {
    if (label == 1.0f) {
      weight *= scale_pos_weight;
    }
    return {(predt - label) * weight, weight};
  }
};

using SquaredErrorGradientKernel = elementwise::GradientKernel<SquaredErrorGradient>;
}  // namespace xgboost::obj

#endif  // XGBOOST_OBJECTIVE_SQUARED_ERROR_OBJ_H_
