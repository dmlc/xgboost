/**
 * Copyright 2026, XGBoost Contributors
 * \file elementwise_objective.h
 * \brief Typed elementwise objective kernels and CPU implementations.
 */
#ifndef XGBOOST_OBJECTIVE_ELEMENTWISE_OBJECTIVE_H_
#define XGBOOST_OBJECTIVE_ELEMENTWISE_OBJECTIVE_H_

#include <algorithm>  // for all_of
#include <cstddef>    // for size_t

#include "../common/kernel.h"            // for KernelRegistration
#include "../common/linalg_op.h"         // for ElementWiseKernel
#include "../common/optional_weight.h"   // for OptionalWeights
#include "../common/threading_utils.h"   // for ParallelFor
#include "xgboost/base.h"                // for GradientPair, bst_target_t
#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/linalg.h"              // for Matrix

namespace xgboost::obj::elementwise {
template <typename GradientFn>
struct GradientKernel {
  using Signature = void(Context const*, HostDeviceVector<float> const&, MetaInfo const&,
                         bst_target_t, GradientFn, linalg::Matrix<GradientPair>*);
};

template <typename TransformFn>
struct TransformKernel {
  using Signature = void(Context const*, HostDeviceVector<float>*, TransformFn);
};

template <typename CheckFn>
struct ValidationKernel {
  using Signature = bool(Context const*, linalg::Matrix<float> const&, CheckFn);
};

namespace detail {
template <typename GradientFn>
void GradientCpu(Context const* ctx, HostDeviceVector<float> const& preds, MetaInfo const& info,
                 bst_target_t n_targets, GradientFn gradient,
                 linalg::Matrix<GradientPair>* out_gpair) {
  auto device = DeviceOrd::CPU();
  auto predt = linalg::MakeTensorView(device, preds.ConstHostSpan(), info.num_row_, n_targets);
  auto labels = info.labels.HostView();
  common::OptionalWeights weights{info.weights_.ConstHostSpan()};

  out_gpair->SetDevice(device);
  out_gpair->Reshape(info.num_row_, n_targets);
  auto gpair = out_gpair->HostView();

  linalg::cpu_impl::ElementWiseKernel(
      gpair, ctx->Threads(), [=](std::size_t i, std::size_t j) mutable {
        gpair(i, j) = gradient(predt(i, j), labels(i, j), weights[i]);
      });
}

template <typename TransformFn>
void TransformCpu(Context const* ctx, HostDeviceVector<float>* preds, TransformFn transform) {
  auto values = preds->HostSpan();
  common::ParallelFor(values.size(), ctx->Threads(),
                      [=](std::size_t i) { values[i] = transform(values[i]); });
}

template <typename CheckFn>
bool ValidationCpu(Context const*, linalg::Matrix<float> const& values, CheckFn check) {
  auto view = values.HostView();
  return std::all_of(linalg::cbegin(view), linalg::cend(view), check);
}
}  // namespace detail

template <typename GradientFn>
auto RegisterGradientCpu() {
  using Kernel = GradientKernel<GradientFn>;
  return common::KernelRegistration<Kernel>{DeviceOrd::kCPU, &detail::GradientCpu<GradientFn>};
}

template <typename TransformFn>
auto RegisterTransformCpu() {
  using Kernel = TransformKernel<TransformFn>;
  return common::KernelRegistration<Kernel>{DeviceOrd::kCPU, &detail::TransformCpu<TransformFn>};
}

template <typename CheckFn>
auto RegisterValidationCpu() {
  using Kernel = ValidationKernel<CheckFn>;
  return common::KernelRegistration<Kernel>{DeviceOrd::kCPU, &detail::ValidationCpu<CheckFn>};
}
}  // namespace xgboost::obj::elementwise

#endif  // XGBOOST_OBJECTIVE_ELEMENTWISE_OBJECTIVE_H_
