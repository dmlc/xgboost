/**
 * Copyright 2026, XGBoost Contributors
 * \file logistic_obj.cu
 * \brief CUDA implementations of logistic objective kernels.
 */
#include <dmlc/registry.h>

#include "elementwise_objective.cuh"
#include "logistic_obj.h"

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(logistic_kernel_cuda);
namespace {
auto const kRegisterLogisticGradientCuda = elementwise::RegisterGradientCuda<LogisticGradient>();
auto const kRegisterLogisticPredTransformCuda =
    elementwise::RegisterTransformCuda<LogisticPredTransform>();
auto const kRegisterLogisticProbToMarginCuda =
    elementwise::RegisterTransformCuda<LogisticProbToMargin>();
auto const kRegisterLogisticValidationCuda =
    elementwise::RegisterValidationCuda<LogisticLabelCheck>();
}  // namespace
}  // namespace xgboost::obj
