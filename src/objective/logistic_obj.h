/**
 * Copyright 2026, XGBoost Contributors
 * \file logistic_obj.h
 * \brief Shared declarations for logistic objectives.
 */
#ifndef XGBOOST_OBJECTIVE_LOGISTIC_OBJ_H_
#define XGBOOST_OBJECTIVE_LOGISTIC_OBJ_H_

#include <cmath>  // for fmaxf

#include "../common/common.h"       // for Min, Max
#include "../common/math.h"         // for Logit, Sigmoid
#include "elementwise_objective.h"  // for elementwise kernels
#include "xgboost/base.h"           // for GradientPair

namespace xgboost::obj {
struct LogisticGradient {
  float scale_pos_weight;

  XGBOOST_DEVICE GradientPair operator()(float predt, float label, float weight) const {
    auto prediction = common::Sigmoid(predt);
    if (label == 1.0f) {
      weight *= scale_pos_weight;
    }
    auto hess = fmaxf(prediction * (1.0f - prediction), 1e-16f);
    return {(prediction - label) * weight, hess * weight};
  }
};

struct LogisticPredTransform {
  XGBOOST_DEVICE float operator()(float value) const { return common::Sigmoid(value); }
};
struct LogisticProbToMargin {
  XGBOOST_DEVICE float operator()(float value) const {
    value = common::Min(common::Max(value, kRtEps), 1.0f - kRtEps);
    return common::Logit(value);
  }
};
struct LogisticLabelCheck {
  XGBOOST_DEVICE bool operator()(float value) const { return value >= 0.0f && value <= 1.0f; }
};

using LogisticGradientKernel = elementwise::GradientKernel<LogisticGradient>;
using LogisticPredTransformKernel = elementwise::TransformKernel<LogisticPredTransform>;
using LogisticProbToMarginKernel = elementwise::TransformKernel<LogisticProbToMargin>;
using LogisticValidationKernel = elementwise::ValidationKernel<LogisticLabelCheck>;
}  // namespace xgboost::obj

#endif  // XGBOOST_OBJECTIVE_LOGISTIC_OBJ_H_
