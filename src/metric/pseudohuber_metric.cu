/**
 * Copyright 2026, XGBoost Contributors
 * \file pseudohuber_metric.cu
 * \brief CUDA implementation of the mean pseudo-Huber error metric kernel.
 */
#include <dmlc/registry.h>

#include "elementwise_metric.cuh"
#include "pseudohuber_metric.h"

namespace xgboost::metric {
DMLC_REGISTRY_FILE_TAG(pseudohuber_metric_cuda);
namespace {
auto const kRegisterPseudoHuberCuda = elementwise::RegisterEvalCuda<EvalRowPseudoHuber>();
}  // namespace
}  // namespace xgboost::metric
