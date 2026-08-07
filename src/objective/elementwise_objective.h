/**
 * Copyright 2026, XGBoost Contributors
 * \file elementwise_objective.h
 * \brief Typed elementwise objective kernels and CPU implementations.
 */
#ifndef XGBOOST_OBJECTIVE_ELEMENTWISE_OBJECTIVE_H_
#define XGBOOST_OBJECTIVE_ELEMENTWISE_OBJECTIVE_H_

#include <cstddef>  // for size_t

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
}  // namespace xgboost::obj::elementwise

#endif  // XGBOOST_OBJECTIVE_ELEMENTWISE_OBJECTIVE_H_
