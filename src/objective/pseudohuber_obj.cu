/**
 * Copyright 2026, XGBoost Contributors
 * \file pseudohuber_obj.cu
 * \brief CUDA implementation of the pseudo-Huber objective.
 */
#include <dmlc/registry.h>

#include "elementwise_objective.cuh"  // for elementwise::RegisterGradientCuda
#include "pseudohuber_obj.h"

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(pseudohuber_kernel_cuda);

namespace {
auto const kRegisterPseudoHuberGradientCuda =
    elementwise::RegisterGradientCuda<PseudoHuberGradient>();
}  // namespace
}  // namespace xgboost::obj
