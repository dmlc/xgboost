/**
 * Copyright 2026, XGBoost Contributors
 * \file poisson_metric.cu
 * \brief CUDA implementation of the Poisson negative log-likelihood metric kernel.
 */
#include <dmlc/registry.h>

#include "elementwise_metric.cuh"
#include "poisson_metric.h"

namespace xgboost::metric {
DMLC_REGISTRY_FILE_TAG(poisson_metric_cuda);
namespace {
auto const kRegisterPoissonCuda = elementwise::RegisterEvalCuda<EvalPoissonNegLogLik>();
}  // namespace
}  // namespace xgboost::metric
