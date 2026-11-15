/**
 * Copyright 2026, XGBoost Contributors
 * \file gamma_metric.cc
 * \brief CPU implementation and registration of gamma metrics.
 */
#include "gamma_metric.h"

#include <dmlc/registry.h>

namespace xgboost::metric {
DMLC_REGISTRY_FILE_TAG(gamma_metric);

namespace {
auto const kRegisterGammaDevianceCpu = elementwise::RegisterEvalCpu<EvalGammaDeviance>();
auto const kRegisterGammaNLogLikCpu = elementwise::RegisterEvalCpu<EvalGammaNLogLik>();
}  // namespace

XGBOOST_REGISTER_METRIC(GammaDeviance, "gamma-deviance")
    .describe("Residual deviance for gamma regression.")
    .set_body([](char const*) {
      return new elementwise::EvalEWiseMetric<EvalGammaDeviance, GammaDevianceEvalKernel>();
    });

XGBOOST_REGISTER_METRIC(GammaNLogLik, "gamma-nloglik")
    .describe("Negative log-likelihood for gamma regression.")
    .set_body([](char const*) {
      return new elementwise::EvalEWiseMetric<EvalGammaNLogLik, GammaNLogLikEvalKernel>();
    });
}  // namespace xgboost::metric
