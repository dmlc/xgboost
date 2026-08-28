/**
 * Copyright 2026, XGBoost Contributors
 * \file pseudohuber_obj.h
 * \brief Shared declarations for the pseudo-Huber objective.
 */
#ifndef XGBOOST_OBJECTIVE_PSEUDOHUBER_OBJ_H_
#define XGBOOST_OBJECTIVE_PSEUDOHUBER_OBJ_H_

#include <cmath>  // for sqrtf

#include "elementwise_objective.h"  // for elementwise::GradientKernel
#include "xgboost/base.h"           // for GradientPair

namespace xgboost::obj {
struct PseudoHuberGradient {
  float slope;

  XGBOOST_DEVICE GradientPair operator()(float predt, float label, float weight) const {
    auto z = predt - label;
    auto slope_sq = slope * slope;
    auto z_sq = z * z;
    auto scale_sqrt = sqrtf(1.0f + z_sq / slope_sq);
    auto grad = z / scale_sqrt;
    auto scale = slope_sq + z_sq;
    auto hess = slope_sq / (scale * scale_sqrt);
    return {grad * weight, hess * weight};
  }
};

using PseudoHuberGradientKernel = elementwise::GradientKernel<PseudoHuberGradient>;
}  // namespace xgboost::obj

#endif  // XGBOOST_OBJECTIVE_PSEUDOHUBER_OBJ_H_
