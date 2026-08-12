/**
 * Copyright 2026, XGBoost Contributors
 * \file poisson_obj.h
 * \brief Shared declarations for the Poisson objective.
 */
#ifndef XGBOOST_OBJECTIVE_POISSON_OBJ_H_
#define XGBOOST_OBJECTIVE_POISSON_OBJ_H_

#include <cmath>  // for expf, log

#include "elementwise_objective.h"  // for elementwise kernels
#include "regression_loss.h"        // for PoissonLabel
#include "xgboost/base.h"           // for GradientPair

namespace xgboost::obj {
struct PoissonGradient {
  XGBOOST_DEVICE GradientPair operator()(float predt, float label, float weight) const {
    auto mu = expf(predt);
    auto grad = (mu - label) * weight;
    // For one leaf, let M = sum(w_i * mu_i) and Y = sum(w_i * y_i). The score after
    // adding leaf value d is f(d) = M * exp(d) - Y. At d = 0, f = M - Y and
    // f' = f'' = M, so Halley's root update is
    //   d = -2 * f * f' / (2 * f'^2 - f * f'') = 2 * (Y - M) / (Y + M).
    // Using the positive pseudo-Hessian h_i = w_i * (mu_i + y_i) / 2 makes the
    // unregularized leaf calculation -sum(g_i) / sum(h_i) produce this update. For M, Y > 0,
    // it is also 2 * tanh(log(Y / M) / 2), so it moves toward the exact optimum without crossing.
    auto hess = 0.5f * (mu + label) * weight;
    return {grad, hess};
  }
};

struct PoissonPredTransform {
  XGBOOST_DEVICE float operator()(float value) const { return expf(value); }
};
struct PoissonProbToMargin {
  XGBOOST_DEVICE float operator()(float value) const { return std::log(value); }
};
struct PoissonLabelCheck {
  XGBOOST_DEVICE bool operator()(float value) const { return PoissonLabel::CheckLabel(value); }
};

using PoissonGradientKernel = elementwise::GradientKernel<PoissonGradient>;
using PoissonPredTransformKernel = elementwise::TransformKernel<PoissonPredTransform>;
using PoissonProbToMarginKernel = elementwise::TransformKernel<PoissonProbToMargin>;
using PoissonValidationKernel = elementwise::ValidationKernel<PoissonLabelCheck>;
}  // namespace xgboost::obj

#endif  // XGBOOST_OBJECTIVE_POISSON_OBJ_H_
