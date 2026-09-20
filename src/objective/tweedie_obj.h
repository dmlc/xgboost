/**
 * Copyright 2026, XGBoost Contributors
 * \file tweedie_obj.h
 * \brief Shared declarations for the Tweedie objective.
 */
#ifndef XGBOOST_OBJECTIVE_TWEEDIE_OBJ_H_
#define XGBOOST_OBJECTIVE_TWEEDIE_OBJ_H_

#include <cmath>  // for expf, log

#include "elementwise_objective.h"  // for elementwise kernels
#include "regression_loss.h"        // for TweedieLabel
#include "xgboost/base.h"           // for GradientPair
#include "xgboost/parameter.h"      // for XGBoostParameter

namespace xgboost::obj {
struct TweedieRegressionParam : public XGBoostParameter<TweedieRegressionParam> {
  float tweedie_variance_power;
  DMLC_DECLARE_PARAMETER(TweedieRegressionParam) {
    DMLC_DECLARE_FIELD(tweedie_variance_power)
        .set_range(1.0f, 2.0f)
        .set_default(1.5f)
        .describe("Tweedie variance power. Must be in the range [1, 2).");
  }
};

struct TweedieGradient {
  float rho;
  XGBOOST_DEVICE GradientPair operator()(float predt, float label, float weight) const {
    auto a = label * expf((1.0f - rho) * predt);
    auto b = expf((2.0f - rho) * predt);
    auto grad = (b - a) * weight;
    // For Tweedie loss, the exact gradient is b - a and the exact Hessian is
    // (rho - 1) * a + (2 - rho) * b. Let A = sum(w_i * a_i) and
    // B = sum(w_i * b_i) in a leaf. Its Newton update is
    //
    //   d = (A - B) / ((rho - 1) * A + (2 - rho) * B).
    //
    // The third derivative is -(rho - 1)^2 * a + (2 - rho)^2 * b. Since both terms
    // vary exponentially with the margin, a large Newton update can leave the region
    // where its local quadratic approximation is accurate.
    //
    // Instead, consider XGBoost's leaf update d = -G/H and quadratic gain
    // q = G^2/(2H). For 1 < rho < 2, the exact leaf loss is
    //
    //   L(d) = A * exp((1 - rho) * d) / (rho - 1)
    //        + B * exp((2 - rho) * d) / (2 - rho),
    //
    // with optimum d* = log(A/B). Matching q to the oracle reduction through cubic
    // order near A = B gives H = (rho * A + (3 - rho) * B)/3, implemented row-wise
    // below. The resulting step
    //
    //   d = 3 * (A - B) / (rho * A + (3 - rho) * B)
    //
    // lies between -3/(3 - rho) and 3/rho. It also satisfies
    //
    //   q <= L(0) - L(d) <= L(0) - L(d*),
    //
    // so the quadratic gain is a conservative estimate of the realized reduction.
    auto hess = (rho * a + (3.0f - rho) * b) * weight / 3.0f;
    return {grad, hess};
  }
};

struct TweediePredTransform {
  XGBOOST_DEVICE float operator()(float value) const { return expf(value); }
};
struct TweedieProbToMargin {
  XGBOOST_DEVICE float operator()(float value) const { return std::log(value); }
};
struct TweedieLabelCheck {
  XGBOOST_DEVICE bool operator()(float value) const { return TweedieLabel::CheckLabel(value); }
};

using TweedieGradientKernel = elementwise::GradientKernel<TweedieGradient>;
using TweediePredTransformKernel = elementwise::TransformKernel<TweediePredTransform>;
using TweedieProbToMarginKernel = elementwise::TransformKernel<TweedieProbToMargin>;
using TweedieValidationKernel = elementwise::ValidationKernel<TweedieLabelCheck>;
}  // namespace xgboost::obj

#endif  // XGBOOST_OBJECTIVE_TWEEDIE_OBJ_H_
