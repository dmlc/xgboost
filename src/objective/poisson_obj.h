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
    // This is the rho=1 endpoint of the bounded-step Tweedie curvature. For a leaf,
    // M = sum(w_i * mu_i) and Y = sum(w_i * y_i). Among affine, row-additive
    // curvatures c*Y + (1-c)*M, matching the exact optimized leaf gain through cubic
    // order gives c=1/3. The resulting quadratic node and split gains lower-bound the
    // realized unregularized full-step loss reduction.
    auto hess = (2.0f * mu + label) * weight / 3.0f;
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
