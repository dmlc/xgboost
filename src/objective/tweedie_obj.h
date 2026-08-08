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
    auto grad = -label * expf((1 - rho) * predt) + expf((2 - rho) * predt);
    auto hess =
        -label * (1 - rho) * std::exp((1 - rho) * predt) + (2 - rho) * expf((2 - rho) * predt);
    return {grad * weight, hess * weight};
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
