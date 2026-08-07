/**
 * Copyright 2026, XGBoost Contributors
 * \file pseudohuber_obj.cu
 * \brief CUDA implementation of the pseudo-Huber objective.
 */
#include <dmlc/registry.h>

#include <cstddef>  // for size_t

#include "../common/kernel.h"           // for KernelRegistration
#include "../common/linalg_op.cuh"      // for ElementWiseKernel
#include "../common/optional_weight.h"  // for MakeOptionalWeights
#include "pseudohuber_obj.h"

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(pseudohuber_kernel_cuda);

namespace cuda_impl {
void PseudoHuberGradient(Context const* ctx, HostDeviceVector<float> const& preds,
                         MetaInfo const& info, bst_target_t n_targets, float slope,
                         linalg::Matrix<GradientPair>* out_gpair) {
  auto device = ctx->Device();
  CHECK(device.IsCUDA());

  preds.SetDevice(device);
  auto predt = linalg::MakeTensorView(ctx, &preds, info.num_row_, n_targets);
  auto labels = info.labels.View(device);
  auto weights = common::MakeOptionalWeights(device, info.weights_);

  out_gpair->SetDevice(device);
  out_gpair->Reshape(info.num_row_, n_targets);
  auto gpair = out_gpair->View(device);

  linalg::cuda_impl::ElementWiseKernel(
      gpair,
      [=] XGBOOST_DEVICE(std::size_t i, std::size_t j) mutable {
        gpair(i, j) = PseudoHuberLoss::Gradient(predt(i, j), labels(i, j), weights[i], slope);
      },
      ctx->CUDACtx()->Stream());
}
}  // namespace cuda_impl

namespace {
common::KernelRegistration<PseudoHuberGradientKernel> const kRegisterPseudoHuberGradientCuda{
    DeviceOrd::kCUDA, &cuda_impl::PseudoHuberGradient};
}  // namespace
}  // namespace xgboost::obj
