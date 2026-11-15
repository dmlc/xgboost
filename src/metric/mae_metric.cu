/**
 * Copyright 2026, XGBoost Contributors
 * \file mae_metric.cu
 * \brief CUDA implementations of the MAE and MAPE metric kernels.
 */
#include <dmlc/registry.h>

#include "elementwise_metric.cuh"
#include "mae_metric.h"

namespace xgboost::metric {
DMLC_REGISTRY_FILE_TAG(mae_metric_cuda);
namespace {
auto const kRegisterMAECuda = elementwise::RegisterEvalCuda<EvalRowMAE>();
auto const kRegisterMAPECuda = elementwise::RegisterEvalCuda<EvalRowMAPE>();
}  // namespace
}  // namespace xgboost::metric
