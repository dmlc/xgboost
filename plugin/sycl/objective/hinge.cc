/**
 * Copyright 2026, XGBoost Contributors
 * \file hinge.cc
 * \brief SYCL implementation of the hinge loss objective kernels.
 */
#include "../../../src/objective/hinge.h"  // for HingeLoss, HingeLabelCheck

#include <dmlc/registry.h>

#include "elementwise_objective.h"

namespace xgboost::sycl::obj {
DMLC_REGISTRY_FILE_TAG(hinge_kernel_sycl);

namespace {
auto const kRegisterHingeGradient =
    xgboost::sycl::obj::elementwise::RegisterGradient<xgboost::obj::HingeLoss>();
auto const kRegisterHingePredTransform =
    xgboost::sycl::obj::elementwise::RegisterTransform<xgboost::obj::HingeLoss>();
auto const kRegisterHingeValidation =
    xgboost::sycl::obj::elementwise::RegisterValidation<xgboost::obj::HingeLabelCheck>();
}  // namespace
}  // namespace xgboost::sycl::obj
