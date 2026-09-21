/**
 * Copyright 2026, XGBoost Contributors
 * \file error_metric.cu
 * \brief CUDA implementation of the classification error metric kernel.
 */
#include <dmlc/registry.h>

#include "elementwise_metric.cuh"
#include "error_metric.h"

namespace xgboost::metric {
DMLC_REGISTRY_FILE_TAG(error_metric_cuda);
namespace {
auto const kRegisterErrorCuda = elementwise::RegisterEvalCuda<EvalError>();
}  // namespace
}  // namespace xgboost::metric
