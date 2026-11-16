/**
 * Copyright 2026, XGBoost Contributors
 */
#include "../common/kernel.h"       // for KernelRegistration
#include "../common/linalg_op.cuh"  // for ElementWiseKernel
#include "random_projection.h"

namespace xgboost::gbm {
namespace {
void ProjectGradient(Context const* ctx, linalg::MatrixView<GradientPair const> in,
                     linalg::MatrixView<double const> basis, linalg::MatrixView<GradientPair> out) {
  linalg::cuda_impl::ElementWiseKernel(out, ProjectGradientKernel{in, basis, out},
                                       ctx->CUDACtx()->Stream());
}

common::KernelRegistration<ProjectGradientKernel> const kProjectGradientCUDA{DeviceOrd::kCUDA,
                                                                             &ProjectGradient};
}  // namespace
}  // namespace xgboost::gbm
