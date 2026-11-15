/**
 * Copyright 2026, XGBoost Contributors
 * \file tweedie_metric.cc
 * \brief CPU implementation and registration of tweedie metrics.
 */
#include "tweedie_metric.h"

#include <dmlc/registry.h>

namespace xgboost::metric {
DMLC_REGISTRY_FILE_TAG(tweedie_metric);

namespace {
auto const kRegisterTweedieNLogLikCpu = elementwise::RegisterEvalCpu<EvalTweedieNLogLik>();
}  // namespace

XGBOOST_REGISTER_METRIC(TweedieNLogLik, "tweedie-nloglik")
    .describe("tweedie-nloglik@rho for tweedie regression.")
    .set_body([](char const* param) {
      return new elementwise::EvalEWiseMetric<EvalTweedieNLogLik, TweedieNLogLikEvalKernel>(param);
    });
}  // namespace xgboost::metric
