/**
 * Copyright 2026, XGBoost Contributors
 * \file gamma_obj.cu
 * \brief CUDA implementations of the gamma objective kernels.
 */
#include <dmlc/registry.h>

#include "elementwise_objective.cuh"
#include "gamma_obj.h"

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(gamma_kernel_cuda);
namespace {
auto const kRegisterGammaGradientCuda = elementwise::RegisterGradientCuda<GammaGradient>();
auto const kRegisterGammaPredTransformCuda =
    elementwise::RegisterTransformCuda<GammaPredTransform>();
auto const kRegisterGammaProbToMarginCuda = elementwise::RegisterTransformCuda<GammaProbToMargin>();
auto const kRegisterGammaValidationCuda = elementwise::RegisterValidationCuda<GammaLabelCheck>();
}  // namespace
}  // namespace xgboost::obj
