/**
 * Copyright 2026, XGBoost Contributors
 * \file expectile_metric.cu
 * \brief CUDA implementation of the expectile metric kernel.
 */
#include <dmlc/registry.h>

#include "alpha_metric.cuh"
#include "expectile_metric.h"

namespace xgboost::metric {
DMLC_REGISTRY_FILE_TAG(expectile_metric_cuda);
namespace {
auto const kRegisterExpectileCuda = alpha::RegisterEvalCuda<EvalRowExpectile>();
}  // namespace
}  // namespace xgboost::metric
