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
  float scale_pos_weight;

  XGBOOST_DEVICE GradientPair operator()(float predt, float label, float weight) const {
    if (label == 1.0f) {
      weight *= scale_pos_weight;
    }
    auto prediction = expf(predt);
    auto ratio = label / prediction;
    auto grad = 1.0f - ratio;
    // This is the rho=2 endpoint of the bounded-step Tweedie curvature. It retains
    // the exact gradient while matching the exact optimized leaf gain through cubic
    // order with an affine, row-additive curvature.
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
