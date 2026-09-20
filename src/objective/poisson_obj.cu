/**
 * Copyright 2026, XGBoost Contributors
 * \file poisson_obj.cu
 * \brief CUDA implementations of the Poisson objective kernels.
 */
#include <dmlc/registry.h>

#include "elementwise_objective.cuh"
#include "poisson_obj.h"

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(poisson_kernel_cuda);
namespace {
auto const kRegisterPoissonGradientCuda = elementwise::RegisterGradientCuda<PoissonGradient>();
auto const kRegisterPoissonPredTransformCuda =
    elementwise::RegisterTransformCuda<PoissonPredTransform>();
auto const kRegisterPoissonProbToMarginCuda =
    elementwise::RegisterTransformCuda<PoissonProbToMargin>();
auto const kRegisterPoissonValidationCuda =
    elementwise::RegisterValidationCuda<PoissonLabelCheck>();
}  // namespace
}  // namespace xgboost::obj
