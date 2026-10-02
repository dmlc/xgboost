/**
 * Copyright 2026, XGBoost Contributors
 * \file elementwise_objective.h
 * \brief SYCL implementations of the typed elementwise objective kernels.
 */
#ifndef PLUGIN_SYCL_OBJECTIVE_ELEMENTWISE_OBJECTIVE_H_
#define PLUGIN_SYCL_OBJECTIVE_ELEMENTWISE_OBJECTIVE_H_

#include <cstddef>  // for size_t

#include "../../../src/common/kernel.h"           // for KernelRegistration
#include "../../../src/common/optional_weight.h"  // for MakeOptionalWeights
#include "../../../src/objective/elementwise_objective.h"
#include "../common/linalg_op.h"  // for ElementWiseKernel, Validate

namespace xgboost::sycl::obj::elementwise {
namespace detail {
template <typename GradientFn>
void GradientSycl(Context const* ctx, HostDeviceVector<float> const& preds, MetaInfo const& info,
                  bst_target_t n_targets, GradientFn gradient,
                  xgboost::linalg::Matrix<GradientPair>* out_gpair) {
  auto device = ctx->Device();
  CHECK(device.IsSycl());

  preds.SetDevice(device);
  auto predt = xgboost::linalg::MakeTensorView(ctx, &preds, info.num_row_, n_targets);
  auto labels = info.labels.View(device);
  auto weights = xgboost::common::MakeOptionalWeights(device, info.weights_);

  out_gpair->SetDevice(device);
  out_gpair->Reshape(info.num_row_, n_targets);
  auto gpair = out_gpair->View(device);

  linalg::ElementWiseKernel(gpair, [=](std::size_t i, std::size_t j) mutable {
    gpair(i, j) = gradient(predt(i, j), labels(i, j), weights[i]);
  });
}

template <typename TransformFn>
void TransformSycl(Context const* ctx, HostDeviceVector<float>* preds, TransformFn transform) {
  auto device = ctx->Device();
  CHECK(device.IsSycl());

  preds->SetDevice(device);
  auto values = xgboost::linalg::MakeTensorView(device, preds->DeviceSpan(), preds->Size());

  linalg::ElementWiseKernel(values,
                            [=](std::size_t i) mutable { values(i) = transform(values(i)); });
}

template <typename CheckFn>
bool ValidationSycl(Context const* ctx, xgboost::linalg::Matrix<float> const& values,
                    CheckFn check) {
  auto device = ctx->Device();
  CHECK(device.IsSycl());
  return linalg::Validate(device, values.View(device), check);
}
}  // namespace detail

template <typename GradientFn>
auto RegisterGradientSycl() {
  using Kernel = xgboost::obj::elementwise::GradientKernel<GradientFn>;
  return xgboost::common::KernelRegistration<Kernel>{
      {DeviceOrd::kSyclDefault, DeviceOrd::kSyclCPU, DeviceOrd::kSyclGPU},
      &detail::GradientSycl<GradientFn>};
}

template <typename TransformFn>
auto RegisterTransformSycl() {
  using Kernel = xgboost::obj::elementwise::TransformKernel<TransformFn>;
  return xgboost::common::KernelRegistration<Kernel>{
      {DeviceOrd::kSyclDefault, DeviceOrd::kSyclCPU, DeviceOrd::kSyclGPU},
      &detail::TransformSycl<TransformFn>};
}

template <typename CheckFn>
auto RegisterValidationSycl() {
  using Kernel = xgboost::obj::elementwise::ValidationKernel<CheckFn>;
  return xgboost::common::KernelRegistration<Kernel>{
      {DeviceOrd::kSyclDefault, DeviceOrd::kSyclCPU, DeviceOrd::kSyclGPU},
      &detail::ValidationSycl<CheckFn>};
}
}  // namespace xgboost::sycl::obj::elementwise

#endif  // PLUGIN_SYCL_OBJECTIVE_ELEMENTWISE_OBJECTIVE_H_
