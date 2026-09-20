/**
 * Copyright 2026, XGBoost Contributors
 * \file gamma_obj.h
 * \brief Shared declarations for the gamma objective.
 */
#ifndef XGBOOST_OBJECTIVE_GAMMA_OBJ_H_
#define XGBOOST_OBJECTIVE_GAMMA_OBJ_H_

#include <cmath>  // for expf, log

#include "elementwise_objective.h"  // for elementwise kernels
#include "xgboost/base.h"           // for GradientPair

namespace xgboost::obj {
struct GammaGradient {
  XGBOOST_DEVICE GradientPair operator()(float predt, float label, float weight) const {
    auto prediction = expf(predt);
    auto ratio = label / prediction;
    auto grad = 1.0f - ratio;
    // For Gamma loss, the exact gradient is 1 - y/mu and the exact Hessian is y/mu.
    // Let A = sum(w_i * y_i/mu_i) and W = sum(w_i) in a leaf. The Newton update
    // (A - W) / A tends to -infinity as A tends to zero.
    //
    // The third derivative is -y/mu, so after a margin update d the Hessian becomes
    // (y/mu) * exp(-d). A large negative Newton update can therefore leave the region
    // where its local quadratic approximation is accurate.
    //
    // Instead, consider XGBoost's leaf update d = -G/H and quadratic gain
    // q = G^2/(2H). The exact leaf loss is L(d) = A * exp(-d) + W * d, with optimum
    // d* = log(A/W). Matching q to the oracle reduction through cubic order near
    // A = W gives H = (2A + W)/3, implemented row-wise by
    // h = (2 * y/mu + 1)/3. The resulting step
    //
    //   d = 3 * (A - W) / (2 * A + W)
    //
    // lies between -3 and 3/2. It also satisfies
    //
    //   q <= L(0) - L(d) <= L(0) - L(d*),
    //
    // so the quadratic gain is a conservative estimate of the realized reduction.
    auto hess = (2.0f * ratio + 1.0f) / 3.0f;
    return {grad * weight, hess * weight};
  }
};

struct GammaPredTransform {
  XGBOOST_DEVICE float operator()(float value) const { return expf(value); }
};
struct GammaProbToMargin {
  XGBOOST_DEVICE float operator()(float value) const { return std::log(value); }
};
struct GammaLabelCheck {
  XGBOOST_DEVICE bool operator()(float value) const { return value > 0.0f; }
};

using GammaGradientKernel = elementwise::GradientKernel<GammaGradient>;
using GammaPredTransformKernel = elementwise::TransformKernel<GammaPredTransform>;
using GammaProbToMarginKernel = elementwise::TransformKernel<GammaProbToMargin>;
using GammaValidationKernel = elementwise::ValidationKernel<GammaLabelCheck>;
}  // namespace xgboost::obj

#endif  // XGBOOST_OBJECTIVE_GAMMA_OBJ_H_
