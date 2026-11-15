/**
 * Copyright 2026, XGBoost Contributors
 * \file pseudohuber_obj.h
 * \brief Shared declarations for the pseudo-Huber objective.
 */
#ifndef XGBOOST_OBJECTIVE_PSEUDOHUBER_OBJ_H_
#define XGBOOST_OBJECTIVE_PSEUDOHUBER_OBJ_H_

#include <cmath>  // for fabsf, hypotf

#include "elementwise_objective.h"  // for elementwise::GradientKernel
#include "xgboost/base.h"           // for GradientPair

namespace xgboost::obj {
struct PseudoHuberGradient {
  float slope;

  XGBOOST_DEVICE GradientPair operator()(float predt, float label, float weight) const {
    auto z = predt - label;
    // The exact Hessian is (1 + (z / slope)^2)^(-3/2). Its Newton step is
    // -z * (1 + (z / slope)^2), which grows cubically for large residuals.
    //
    // Concavity of sqrt as a function of z^2 gives a touching quadratic upper bound
    // with curvature (1 + (z / slope)^2)^(-1/2). Keep the exact gradient and minimize
    // this bound instead: a single-row step is -z, and a leaf step minimizes the sum
    // of the row bounds. Shrinking that step by eta in [0, 1] preserves loss descent
    // for a fixed leaf without regularization.
    // hypot avoids overflow from explicitly squaring a large residual.
    auto hess = fabsf(slope) / hypotf(slope, z);
    auto grad = z * hess;
    return {grad * weight, hess * weight};
  }
};

using PseudoHuberGradientKernel = elementwise::GradientKernel<PseudoHuberGradient>;
}  // namespace xgboost::obj

#endif  // XGBOOST_OBJECTIVE_PSEUDOHUBER_OBJ_H_
