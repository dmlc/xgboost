/**
 * Copyright 2026, XGBoost Contributors
 * \file squared_log_obj.cu
 * \brief CUDA implementations of the squared-log objective kernels.
 */
#include <dmlc/registry.h>

#include "elementwise_objective.cuh"
#include "squared_log_obj.h"

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(squared_log_kernel_cuda);
namespace {
auto const kRegisterSquaredLogGradientCuda =
    elementwise::RegisterGradientCuda<SquaredLogGradient>();
auto const kRegisterSquaredLogValidationCuda =
    elementwise::RegisterValidationCuda<SquaredLogLabelCheck>();
}  // namespace
}  // namespace xgboost::obj
