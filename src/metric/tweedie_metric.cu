/**
 * Copyright 2026, XGBoost Contributors
 * \file tweedie_metric.cu
 * \brief CUDA implementations of tweedie metric kernels.
 */
#include <dmlc/registry.h>

#include "elementwise_metric.cuh"
#include "tweedie_metric.h"

namespace xgboost::metric {
DMLC_REGISTRY_FILE_TAG(tweedie_metric_cuda);
namespace {
auto const kRegisterTweedieNLogLikCuda = elementwise::RegisterEvalCuda<EvalTweedieNLogLik>();
}  // namespace
}  // namespace xgboost::metric
