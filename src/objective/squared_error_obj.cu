/**
 * Copyright 2026, XGBoost Contributors
 * \file squared_error_obj.cu
 * \brief CUDA implementation of the squared-error objective kernel.
 */
#include <dmlc/registry.h>

#include "elementwise_objective.cuh"
#include "squared_error_obj.h"

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(squared_error_kernel_cuda);
namespace {
auto const kRegisterSquaredErrorGradientCuda =
    elementwise::RegisterGradientCuda<SquaredErrorGradient>();
}  // namespace
}  // namespace xgboost::obj
