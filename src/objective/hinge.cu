/**
 * Copyright 2018-2026, XGBoost Contributors
 * \file hinge.cu
 * \brief CUDA implementation of the hinge loss objective.
 * \author Henry Gouk
 */
#include <dmlc/registry.h>

#include <cstddef>  // for size_t

#include "../common/device_helpers.cuh"  // for LaunchN
#include "../common/kernel.h"            // for KernelRegistration
#include "../common/linalg_op.cuh"       // for ElementWiseKernel
#include "../common/optional_weight.h"   // for OptionalWeights
#include "hinge.h"

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(hinge_kernel_cuda);

namespace cuda_impl {
void HingeGradient(Context const* ctx, HostDeviceVector<float> const& preds, MetaInfo const& info,
                   bst_target_t n_targets, linalg::Matrix<GradientPair>* out_gpair) {
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
        gpair(i, j) = HingeLoss::Gradient(predt(i, j), labels(i, j), weights[i]);
      },
      ctx->CUDACtx()->Stream());
}

void HingePredTransform(Context const* ctx, HostDeviceVector<float>* preds) {
  auto device = ctx->Device();
  CHECK(device.IsCUDA());
  preds->SetDevice(device);
  auto values = preds->DeviceSpan();

  dh::LaunchN(values.size(), ctx->CUDACtx()->Stream(), [=] XGBOOST_DEVICE(std::size_t i) {
    values[i] = HingeLoss::PredTransform(values[i]);
  });
}
}  // namespace cuda_impl

namespace {
common::KernelRegistration<HingeGradientKernel> const register_hinge_gradient_cuda{
    DeviceOrd::kCUDA, &cuda_impl::HingeGradient};
common::KernelRegistration<HingePredTransformKernel> const register_hinge_pred_transform_cuda{
    DeviceOrd::kCUDA, &cuda_impl::HingePredTransform};
}  // namespace
}  // namespace xgboost::obj
