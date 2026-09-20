/**
 * Copyright 2026, XGBoost Contributors
 * \file tweedie_obj.cu
 * \brief CUDA implementations of the Tweedie objective kernels.
 */
#include <dmlc/registry.h>

#include "elementwise_objective.cuh"
#include "tweedie_obj.h"

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(tweedie_kernel_cuda);
namespace {
auto const kRegisterTweedieGradientCuda = elementwise::RegisterGradientCuda<TweedieGradient>();
auto const kRegisterTweediePredTransformCuda =
    elementwise::RegisterTransformCuda<TweediePredTransform>();
auto const kRegisterTweedieProbToMarginCuda =
    elementwise::RegisterTransformCuda<TweedieProbToMargin>();
auto const kRegisterTweedieValidationCuda =
    elementwise::RegisterValidationCuda<TweedieLabelCheck>();
}  // namespace
}  // namespace xgboost::obj
