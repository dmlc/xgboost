/**
 * Copyright 2026, XGBoost Contributors
 * \file poisson_obj.h
 * \brief Shared declarations for the Poisson objective.
 */
#ifndef XGBOOST_OBJECTIVE_POISSON_OBJ_H_
#define XGBOOST_OBJECTIVE_POISSON_OBJ_H_

#include <cmath>  // for expf, fmaxf, log

#include "elementwise_objective.h"  // for elementwise kernels
#include "regression_loss.h"        // for PoissonLabel
#include "xgboost/base.h"           // for GradientPair

namespace xgboost::obj {
struct PoissonGradient {
  XGBOOST_DEVICE GradientPair operator()(float predt, float label, float weight) const {
    auto mu = expf(predt);
    auto grad = (mu - label) * weight;
    // For one leaf, let M = sum(w_i * mu_i), Y = sum(w_i * y_i), and
    // H = sum(w_i * max(mu_i, y_i)). For non-negative row weights,
    // H >= max(M, Y), so the unregularized update d = (Y - M) / H obeys,
    // for M, Y > 0:
    //   Y >= M: 0 <= d <= 1 - M / Y <= log(Y / M)
    //   Y <= M: log(Y / M) <= Y / M - 1 <= d <= 0.
    // Thus 0 <= eta <= 1 moves toward the exact leaf optimum log(Y / M)
    // without crossing it. L1/L2 regularization can only reduce |d|.
    auto hess = fmaxf(mu, label) * weight;
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
