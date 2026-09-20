/**
 * Copyright 2026, XGBoost Contributors
 * \file poisson_metric.cc
 * \brief CPU implementation and registration of the Poisson negative log-likelihood metric.
 */
#include "poisson_metric.h"

#include <dmlc/registry.h>

namespace xgboost::metric {
DMLC_REGISTRY_FILE_TAG(poisson_metric);

namespace {
auto const kRegisterPoissonCpu = elementwise::RegisterEvalCpu<EvalPoissonNegLogLik>();
}  // namespace

XGBOOST_REGISTER_METRIC(PossionNegLoglik, "poisson-nloglik")
    .describe("Negative loglikelihood for poisson regression.")
    .set_body([](char const*) {
      return new elementwise::EvalEWiseMetric<EvalPoissonNegLogLik, PoissonEvalKernel>();
    });
}  // namespace xgboost::metric
