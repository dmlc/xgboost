/**
 * Copyright 2026, XGBoost Contributors
 * \file logloss_metric.cu
 * \brief CUDA implementation of the log loss metric kernel.
 */
#include <dmlc/registry.h>

#include "elementwise_metric.cuh"
#include "logloss_metric.h"

namespace xgboost::metric {
DMLC_REGISTRY_FILE_TAG(logloss_metric_cuda);
namespace {
auto const kRegisterLogLossCuda = elementwise::RegisterEvalCuda<EvalRowLogLoss>();
}  // namespace
}  // namespace xgboost::metric
