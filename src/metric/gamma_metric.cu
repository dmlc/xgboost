/**
 * Copyright 2026, XGBoost Contributors
 * \file gamma_metric.cu
 * \brief CUDA implementations of gamma metric kernels.
 */
#include <dmlc/registry.h>

#include "elementwise_metric.cuh"
#include "gamma_metric.h"

namespace xgboost::metric {
DMLC_REGISTRY_FILE_TAG(gamma_metric_cuda);
namespace {
auto const kRegisterGammaDevianceCuda = elementwise::RegisterEvalCuda<EvalGammaDeviance>();
auto const kRegisterGammaNLogLikCuda = elementwise::RegisterEvalCuda<EvalGammaNLogLik>();
}  // namespace
}  // namespace xgboost::metric
