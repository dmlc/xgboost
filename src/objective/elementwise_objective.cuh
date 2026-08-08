/**
 * Copyright 2026, XGBoost Contributors
 * \file elementwise_objective.cuh
 * \brief CUDA implementations of the typed elementwise objective kernels.
 */
#ifndef XGBOOST_OBJECTIVE_ELEMENTWISE_OBJECTIVE_CUH_
#define XGBOOST_OBJECTIVE_ELEMENTWISE_OBJECTIVE_CUH_

#include <cstddef>  // for size_t

#include "../common/algorithm.cuh"       // for AllOf
#include "../common/device_helpers.cuh"  // for LaunchN, MakeIndexTransformIter
#include "../common/linalg_op.cuh"       // for ElementWiseKernel
#include "../common/optional_weight.h"   // for MakeOptionalWeights
#include "elementwise_objective.h"

namespace xgboost::obj::elementwise {
namespace detail {
template <typename GradientFn>
void GradientCuda(Context const* ctx, HostDeviceVector<float> const& preds, MetaInfo const& info,
                  bst_target_t n_targets, GradientFn gradient,
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
        gpair(i, j) = gradient(predt(i, j), labels(i, j), weights[i]);
      },
      ctx->CUDACtx()->Stream());
}

template <typename TransformFn>
void TransformCuda(Context const* ctx, HostDeviceVector<float>* preds, TransformFn transform) {
  auto device = ctx->Device();
  CHECK(device.IsCUDA());
  preds->SetDevice(device);
  auto values = preds->DeviceSpan();

  dh::LaunchN(values.size(), ctx->CUDACtx()->Stream(),
              [=] XGBOOST_DEVICE(std::size_t i) { values[i] = transform(values[i]); });
}

template <typename CheckFn>
bool ValidationCuda(Context const* ctx, linalg::Matrix<float> const& values, CheckFn check) {
  auto device = ctx->Device();
  CHECK(device.IsCUDA());
  auto view = values.View(device);
  auto iter = dh::MakeIndexTransformIter([=] XGBOOST_DEVICE(std::size_t i) {
    auto [m, n] = linalg::UnravelIndex(i, view.Shape());
    return view(m, n);
  });
  return common::AllOf(ctx->CUDACtx()->CTP(), iter, iter + view.Size(), check);
}
}  // namespace detail

template <typename GradientFn>
auto RegisterGradientCuda() {
  using Kernel = GradientKernel<GradientFn>;
  return common::KernelRegistration<Kernel>{DeviceOrd::kCUDA, &detail::GradientCuda<GradientFn>};
}

template <typename TransformFn>
auto RegisterTransformCuda() {
  using Kernel = TransformKernel<TransformFn>;
  return common::KernelRegistration<Kernel>{DeviceOrd::kCUDA, &detail::TransformCuda<TransformFn>};
}

template <typename CheckFn>
auto RegisterValidationCuda() {
  using Kernel = ValidationKernel<CheckFn>;
  return common::KernelRegistration<Kernel>{DeviceOrd::kCUDA, &detail::ValidationCuda<CheckFn>};
}
}  // namespace xgboost::obj::elementwise

#endif  // XGBOOST_OBJECTIVE_ELEMENTWISE_OBJECTIVE_CUH_
