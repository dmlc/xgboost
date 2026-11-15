/**
 * Copyright 2026, XGBoost Contributors
 * \file rmse_metric.cu
 * \brief CUDA implementations of the RMSE and RMSLE metric kernels.
 */
#include <dmlc/registry.h>

#include "elementwise_metric.cuh"
#include "rmse_metric.h"

namespace xgboost::metric {
DMLC_REGISTRY_FILE_TAG(rmse_metric_cuda);
namespace {
auto const kRegisterRMSECuda = elementwise::RegisterEvalCuda<EvalRowRMSE>();
auto const kRegisterRMSLECuda = elementwise::RegisterEvalCuda<EvalRowRMSLE>();
}  // namespace
}  // namespace xgboost::metric
