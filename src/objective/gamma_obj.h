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
#include "xgboost/string_view.h"    // for StringView

namespace xgboost::obj {
class GammaDeviance {
 public:
  constexpr static StringView InterceptErrorMsg() {
    return "`base_score` must be greater than 0 for gamma regression";
  }
  XGBOOST_DEVICE static bool CheckIntercept(float base_score) { return base_score > 0; }
  XGBOOST_DEVICE static float FirstOrderGradient(float predt, float label) {
    return 1.0f - label / predt;
  }
  XGBOOST_DEVICE static float SecondOrderGradient(float predt, float label) {
    return label / predt;
  }
  static char const* Name() { return "reg:gamma"; }
  XGBOOST_DEVICE static bool CheckLabel(float label) { return label > 0.0f; }
  static char const* LabelErrorMsg() { return "label must be positive for gamma regression."; }
};

struct GammaGradient {
  float scale_pos_weight;

  XGBOOST_DEVICE GradientPair operator()(float predt, float label, float weight) const {
    if (label == 1.0f) {
      weight *= scale_pos_weight;
    }
    auto prediction = expf(predt);
    auto grad = GammaDeviance::FirstOrderGradient(prediction, label);
    auto hess = GammaDeviance::SecondOrderGradient(prediction, label);
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
  XGBOOST_DEVICE bool operator()(float value) const { return GammaDeviance::CheckLabel(value); }
};

using GammaGradientKernel = elementwise::GradientKernel<GammaGradient>;
using GammaPredTransformKernel = elementwise::TransformKernel<GammaPredTransform>;
using GammaProbToMarginKernel = elementwise::TransformKernel<GammaProbToMargin>;
using GammaValidationKernel = elementwise::ValidationKernel<GammaLabelCheck>;
}  // namespace xgboost::obj

#endif  // XGBOOST_OBJECTIVE_GAMMA_OBJ_H_
