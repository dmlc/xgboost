/**
 * Copyright 2026, XGBoost Contributors
 * \file quantile_metric.cu
 * \brief CUDA implementation of the quantile metric kernel.
 */
#include <dmlc/registry.h>

#include "alpha_metric.cuh"
#include "quantile_metric.h"

namespace xgboost::metric {
DMLC_REGISTRY_FILE_TAG(quantile_metric_cuda);
namespace {
auto const kRegisterQuantileCuda = alpha::RegisterEvalCuda<EvalRowQuantile>();
}  // namespace
}  // namespace xgboost::metric
