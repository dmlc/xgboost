/**
 * Copyright 2026, XGBoost Contributors
 * \file squared_log_obj.h
 * \brief Shared declarations for the squared-log objective.
 */
#ifndef XGBOOST_OBJECTIVE_SQUARED_LOG_OBJ_H_
#define XGBOOST_OBJECTIVE_SQUARED_LOG_OBJ_H_

#include <cmath>  // for fmaxf, log1p, pow

#include "elementwise_objective.h"  // for elementwise kernels
#include "xgboost/base.h"           // for GradientPair

namespace xgboost::obj {
struct SquaredLogError {
  XGBOOST_DEVICE static bool CheckLabel(float label) { return label > -1.0f; }
  XGBOOST_DEVICE static float FirstOrderGradient(float predt, float label) {
    predt = fmaxf(predt, -1.0f + 1e-6f);
    return (std::log1p(predt) - std::log1p(label)) / (predt + 1.0f);
  }
  XGBOOST_DEVICE static float SecondOrderGradient(float predt, float label) {
    predt = fmaxf(predt, -1.0f + 1e-6f);
    auto hess = (-std::log1p(predt) + std::log1p(label) + 1.0f) / std::pow(predt + 1.0f, 2.0f);
    return fmaxf(hess, 1e-6f);
  }
  static char const* LabelErrorMsg() {
    return "label must be greater than -1 for rmsle so that log(label + 1) can be valid.";
  }
  static char const* Name() { return "reg:squaredlogerror"; }
};

struct SquaredLogGradient {
  XGBOOST_DEVICE GradientPair operator()(float predt, float label, float weight) const {
    auto grad = SquaredLogError::FirstOrderGradient(predt, label);
    auto hess = SquaredLogError::SecondOrderGradient(predt, label);
    return {grad * weight, hess * weight};
  }
};

struct SquaredLogLabelCheck {
  XGBOOST_DEVICE bool operator()(float value) const { return SquaredLogError::CheckLabel(value); }
};

using SquaredLogGradientKernel = elementwise::GradientKernel<SquaredLogGradient>;
using SquaredLogValidationKernel = elementwise::ValidationKernel<SquaredLogLabelCheck>;
}  // namespace xgboost::obj

#endif  // XGBOOST_OBJECTIVE_SQUARED_LOG_OBJ_H_
