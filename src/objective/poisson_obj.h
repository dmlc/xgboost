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
#include "xgboost/parameter.h"      // for XGBoostParameter

namespace xgboost::obj {
struct PoissonRegressionParam : public XGBoostParameter<PoissonRegressionParam> {
  float max_delta_step;
  DMLC_DECLARE_PARAMETER(PoissonRegressionParam) {
    DMLC_DECLARE_FIELD(max_delta_step)
        .set_lower_bound(0.0f)
        .set_default(0.7f)
        .describe(
            "Maximum delta step we allow each weight estimation to be."
            " This parameter is required for possion regression.");
  }
};

struct PoissonGradient {
  float max_delta_step;
  XGBOOST_DEVICE GradientPair operator()(float predt, float label, float weight) const {
    auto grad = (expf(predt) - label) * weight;
    auto hess = expf(predt + max_delta_step) * weight;
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
