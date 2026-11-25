/**
 * Copyright 2026, XGBoost Contributors
 * \file quantile_obj.cc
 * \brief SYCL implementation of the quantile objective kernels.
 */
#include "../../../src/objective/quantile_obj.h"

#include <dmlc/registry.h>

#include <cmath>    // for fabsf, fmaxf, sqrtf, tanhf
#include <cstddef>  // for size_t

#include "../../../src/collective/aggregator.h"        // for GlobalSum
#include "../../../src/collective/communicator-inl.h"  // for IsDistributed
#include "../../../src/common/kernel.h"                // for KernelRegistration
#include "../../../src/common/optional_weight.h"       // for MakeOptionalWeights
#include "../common/linalg_op.h"                       // for ElementWiseKernel
#include "../device_manager.h"                         // for DeviceManager

namespace xgboost::sycl::obj {
DMLC_REGISTRY_FILE_TAG(quantile_kernel_sycl);

namespace {
constexpr float kSmoothingScale{0.04f};
constexpr float kMinSurrogateRatio{3.0e-4f};

void QuantileGradient(Context const* ctx, HostDeviceVector<float> const& preds,
                      MetaInfo const& info, bst_target_t n_targets,
                      HostDeviceVector<float> const& alpha,
                      xgboost::linalg::Matrix<GradientPair>* out_gpair) {
  auto device = ctx->Device();
  CHECK(device.IsSycl());
  preds.SetDevice(device);
  alpha.SetDevice(device);
  auto predt = xgboost::linalg::MakeTensorView(ctx, &preds, info.num_row_, n_targets);
  auto labels = info.labels.View(device);
  auto weights = xgboost::common::MakeOptionalWeights(device, info.weights_);
  auto alpha_d = alpha.ConstDeviceSpan();

  sycl::DeviceManager device_manager;
  auto* qu = device_manager.GetQueue(device);

  auto n_stats = static_cast<std::size_t>(n_targets) + 1;
  auto scale_stats = xgboost::linalg::Zeros<double>(ctx, n_stats);
  auto stats = scale_stats.View(device);
  if (info.num_row_ != 0) {
    auto* stats_ptr = stats.Values().data();
    for (bst_target_t target{0}; target < n_targets; ++target) {
      qu->submit([&](::sycl::handler& cgh) {
        auto reduction = ::sycl::reduction(stats_ptr + target, ::sycl::plus<>());
        cgh.parallel_for<>(::sycl::range<1>(info.num_row_), reduction,
                           [=](::sycl::id<1> pid, auto& sum) {
                             std::size_t i = pid[0];
                             sum += weights[i] * sqrtf(fabsf(predt(i, target) - labels(i, 0)));
                           });
      });
    }
    qu->submit([&](::sycl::handler& cgh) {
      auto reduction = ::sycl::reduction(stats_ptr + n_targets, ::sycl::plus<>());
      cgh.parallel_for<>(::sycl::range<1>(info.num_row_), reduction,
                         [=](::sycl::id<1> pid, auto& sum) { sum += weights[pid[0]]; });
    });
    qu->wait_and_throw();
  }
  if (xgboost::collective::IsDistributed()) {
    // The CPU allreduce can't access device memory; re-fetch the view to sync the result back.
    auto cpu_ctx = ctx->MakeCPU();
    xgboost::collective::SafeColl(xgboost::collective::GlobalSum(&cpu_ctx, scale_stats.HostView()));
    stats = scale_stats.View(device);
  }

  auto scale = stats.Slice(xgboost::linalg::Range(static_cast<bst_target_t>(0), n_targets));
  sycl::linalg::ElementWiseKernel(scale, [=](std::size_t target) mutable {
    auto sum_weight = stats(n_targets);
    if (sum_weight != 0.0) {
      auto root_mean = stats(target) / sum_weight;
      stats(target) = root_mean * root_mean;
    } else {
      stats(target) = 0.0;
    }
  });

  out_gpair->SetDevice(device);
  out_gpair->Reshape(info.num_row_, n_targets);
  auto gpair = out_gpair->View(device);
  sycl::linalg::ElementWiseKernel(gpair, [=](std::size_t i, std::size_t j) mutable {
    auto residual = predt(i, j) - labels(i, 0);
    auto residual_scale = static_cast<float>(stats(j));
    auto weight = weights[i];
    if (!(residual_scale > 0.0f) || weight == 0.0f) {
      gpair(i, j) = GradientPair{0.0f, 0.0f};
      return;
    }

    auto x = residual / (kSmoothingScale * residual_scale);
    auto tanh_x = tanhf(x);
    auto ratio = x == 0.0f ? 1.0f : tanh_x / x;
    ratio = fmaxf(ratio, kMinSurrogateRatio);
    auto grad = 0.5f * residual_scale * (tanh_x + 1.0f - 2.0f * alpha_d[j]);
    auto hess = 0.5f / kSmoothingScale * ratio;
    gpair(i, j) = GradientPair{weight * grad, weight * hess};
  });
}

void QuantileTransform(Context const* ctx, HostDeviceVector<float>* predictions,
                       std::size_t n_alphas) {
  auto device = ctx->Device();
  CHECK(device.IsSycl());
  predictions->SetDevice(device);
  auto* values = predictions->DevicePointer();
  auto n_rows = predictions->Size() / n_alphas;

  sycl::DeviceManager device_manager;
  auto* qu = device_manager.GetQueue(device);
  qu->submit([&](::sycl::handler& cgh) {
      cgh.parallel_for<>(::sycl::range<1>(n_rows), [=](::sycl::id<1> pid) {
        auto* point = values + pid[0] * n_alphas;
        for (std::size_t i{1}; i < n_alphas; ++i) {
          auto value = point[i];
          auto pos = i;
          while (pos > 0 && point[pos - 1] > value) {
            point[pos] = point[pos - 1];
            --pos;
          }
          point[pos] = value;
        }
      });
    }).wait_and_throw();
}

auto const kRegisterQuantileGradient =
    xgboost::common::KernelRegistration<xgboost::obj::QuantileGradientKernel>{
        {DeviceOrd::kSyclDefault, DeviceOrd::kSyclCPU, DeviceOrd::kSyclGPU},
        &xgboost::sycl::obj::QuantileGradient};
auto const kRegisterQuantileTransform =
    xgboost::common::KernelRegistration<xgboost::obj::QuantileTransformKernel>{
        {DeviceOrd::kSyclDefault, DeviceOrd::kSyclCPU, DeviceOrd::kSyclGPU},
        &xgboost::sycl::obj::QuantileTransform};
}  // namespace
}  // namespace xgboost::sycl::obj
