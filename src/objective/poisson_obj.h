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
    // For Poisson loss, the exact gradient is mu - y and the exact Hessian is mu.
    // Let M = sum(w_i * mu_i) and Y = sum(w_i * y_i) in a leaf. The Newton update
    // (Y - M) / M tends to +infinity as M tends to zero with Y > 0.
    //
    // The third derivative is also mu, so after a margin update d the Hessian becomes
    // mu * exp(d). A large Newton update can therefore leave the region where its
    // local quadratic approximation is accurate.
    //
    // Instead, consider XGBoost's leaf update d = -G/H and quadratic gain
    // q = G^2/(2H). The exact leaf loss is L(d) = M * exp(d) - Y * d, with optimum
    // d* = log(Y/M). Matching q to the oracle reduction L(0) - L(d*) through cubic
    // order near M = Y gives H = (2M + Y)/3, implemented row-wise by
    // h = (2 * mu + y)/3. The resulting step
    //
    //   d = 3 * (Y - M) / (2 * M + Y)
    //
    // is bounded to [-3/2, 3). It also satisfies
    //
    //   q <= L(0) - L(d) <= L(0) - L(d*),
    //
    // so the quadratic gain is a conservative estimate of the realized reduction.
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
