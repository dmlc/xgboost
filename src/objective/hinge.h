/**
 * Copyright 2018-2026, XGBoost Contributors
 * \file hinge.h
 * \brief Shared declarations for the hinge loss objective.
 */
#ifndef XGBOOST_OBJECTIVE_HINGE_H_
#define XGBOOST_OBJECTIVE_HINGE_H_

#include <limits>  // for numeric_limits

#include "elementwise_objective.h"  // for elementwise::GradientKernel, elementwise::TransformKernel
#include "xgboost/base.h"           // for GradientPair

namespace xgboost::obj {
struct HingeLoss {
  XGBOOST_DEVICE GradientPair operator()(float margin, float label, float weight) const {
    auto y = label * 2.0f - 1.0f;
    if (margin * y < 1.0f) {
      return GradientPair{-y * weight, weight};
    }
    return GradientPair{0.0f, std::numeric_limits<float>::min()};
  }

  XGBOOST_DEVICE float operator()(float margin) const { return margin > 0.0f ? 1.0f : 0.0f; }

  static char const* LabelErrorMsg() { return "label must be either 0 or 1 for hinge loss."; }
};

struct HingeLabelCheck {
  XGBOOST_DEVICE bool operator()(float label) const { return label == 0.0f || label == 1.0f; }
};

using HingeGradientKernel = elementwise::GradientKernel<HingeLoss>;
using HingePredTransformKernel = elementwise::TransformKernel<HingeLoss>;
using HingeValidationKernel = elementwise::ValidationKernel<HingeLabelCheck>;

}  // namespace xgboost::obj

#endif  // XGBOOST_OBJECTIVE_HINGE_H_
