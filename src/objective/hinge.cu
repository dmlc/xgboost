/**
 * Copyright 2018-2026, XGBoost Contributors
 * \file hinge.cu
 * \brief CUDA implementation of the hinge loss objective.
 * \author Henry Gouk
 */
#include <dmlc/registry.h>

#include "elementwise_objective.cuh"  // for elementwise kernel registration
#include "hinge.h"

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(hinge_kernel_cuda);

namespace {
auto const kRegisterHingeGradientCuda = elementwise::RegisterGradientCuda<HingeLoss>();
auto const kRegisterHingePredTransformCuda = elementwise::RegisterTransformCuda<HingeLoss>();
}  // namespace
}  // namespace xgboost::obj
