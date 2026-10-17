/**
 * Copyright 2026, XGBoost Contributors
 * \file expectile_obj.cu
 * \brief CUDA implementations of expectile objective kernels.
 */
#include <dmlc/registry.h>

#include <cstddef>

#include "../common/device_helpers.cuh"
#include "../common/expectile_loss_utils.h"
#include "../common/kernel.h"
#include "../common/linalg_op.cuh"
#include "../common/math.h"
#include "../common/optional_weight.h"
#include "../common/stats.h"
#include "../tree/fit_stump.h"
#include "expectile_obj.h"

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(expectile_kernel_cuda);
namespace {
void ExpectileGradientCuda(Context const* ctx, HostDeviceVector<float> const& preds,
                           MetaInfo const& info, HostDeviceVector<float> const& alpha,
                           bst_target_t n_targets, linalg::Matrix<GradientPair>* out_gpair) {
  auto device = ctx->Device();
  CHECK(device.IsCUDA());
  preds.SetDevice(device);
  alpha.SetDevice(device);
  auto labels = info.labels.View(device);
  auto weights = common::MakeOptionalWeights(device, info.weights_);
  auto predt = linalg::MakeTensorView(ctx, &preds, info.num_row_, n_targets);
  auto alpha_d = alpha.ConstDeviceSpan();

  out_gpair->SetDevice(device);
  out_gpair->Reshape(info.num_row_, n_targets);
  auto gpair = out_gpair->View(device);
  linalg::cuda_impl::ElementWiseKernel(
      gpair,
      [=] XGBOOST_DEVICE(std::size_t i, std::size_t j) mutable {
        auto label = labels(i, 0);
        auto sample_weight = weights[i];
        float pred = predt(i, 0);
        float grad_sum{0.0f};
        float hess_sum{0.0f};
        for (std::size_t k{0}; k < alpha_d.size(); ++k) {
          if (k > 0) {
            pred += kRtEps + common::SoftPlus(predt(i, k));
          }
          if (k >= j) {
            auto diff = pred - label;
            auto weight_scale = diff >= 0.0f ? 1.0f - alpha_d[k] : alpha_d[k];
            grad_sum += weight_scale * diff * sample_weight;
            hess_sum += weight_scale * sample_weight;
          }
        }
        auto scale = j == 0 ? 1.0f : common::Sigmoid(predt(i, j));
        gpair(i, j) = {scale * grad_sum, scale * scale * hess_sum};
      },
      ctx->CUDACtx()->Stream());
}

void ExpectileInitEstimationCuda(Context const* ctx, MetaInfo const& info,
                                 HostDeviceVector<float> const& alpha, bst_target_t n_targets,
                                 linalg::Vector<float>* base_score) {
  auto device = ctx->Device();
  CHECK(device.IsCUDA());
  linalg::Vector<float> label_mean;
  if (info.weights_.Empty()) {
    common::SampleMean(ctx, info.labels, &label_mean);
  } else {
    common::WeightedSampleMean(ctx, info.labels, info.weights_, &label_mean);
  }
  CHECK_EQ(label_mean.Size(), 1);
  auto mean = label_mean.View(device);

  alpha.SetDevice(device);
  auto alpha_d = alpha.ConstDeviceSpan();
  auto labels = info.labels.View(device);
  auto weights = common::MakeOptionalWeights(device, info.weights_);
  linalg::Matrix<GradientPair> gpair;
  gpair.SetDevice(device);
  gpair.Reshape(info.num_row_, n_targets);
  auto gpair_d = gpair.View(device);
  linalg::cuda_impl::ElementWiseKernel(
      gpair_d,
      [=] XGBOOST_DEVICE(std::size_t i, std::size_t j) mutable {
        auto diff = mean(0) - labels(i, 0);
        auto weight_scale = diff >= 0.0f ? 1.0f - alpha_d[j] : alpha_d[j];
        gpair_d(i, j) = {weight_scale * diff * weights[i], weight_scale * weights[i]};
      },
      ctx->CUDACtx()->Stream());

  tree::FitStump(ctx, gpair, n_targets, base_score);
  auto out = base_score->View(device);
  dh::LaunchN(1, ctx->CUDACtx()->Stream(), [=] XGBOOST_DEVICE(std::size_t) mutable {
    auto mean_value = mean(0);
    for (std::size_t j{0}; j < n_targets; ++j) {
      out(j) += mean_value;
    }
    for (std::size_t j{1}; j < n_targets; ++j) {
      out(j) = out(j) < out(j - 1) ? out(j - 1) : out(j);
    }
  });
}

void ExpectilePredTransformCuda(Context const* ctx, HostDeviceVector<float>* predictions,
                                std::size_t n_alphas) {
  auto device = ctx->Device();
  CHECK(device.IsCUDA());
  predictions->SetDevice(device);
  auto values = predictions->DeviceSpan();
  auto n_samples = values.size() / n_alphas;
  dh::LaunchN(n_samples, ctx->CUDACtx()->Stream(), [=] XGBOOST_DEVICE(std::size_t i) {
    auto row = values.subspan(i * n_alphas, n_alphas);
    float pred = row[0];
    for (std::size_t j{1}; j < n_alphas; ++j) {
      pred += kRtEps + common::SoftPlus(row[j]);
      row[j] = pred;
    }
  });
}

auto const kRegisterGradientCuda =
    common::KernelRegistration<ExpectileGradientKernel>{DeviceOrd::kCUDA, &ExpectileGradientCuda};
auto const kRegisterInitCuda = common::KernelRegistration<ExpectileInitEstimationKernel>{
    DeviceOrd::kCUDA, &ExpectileInitEstimationCuda};
auto const kRegisterTransformCuda = common::KernelRegistration<ExpectilePredTransformKernel>{
    DeviceOrd::kCUDA, &ExpectilePredTransformCuda};
}  // namespace
}  // namespace xgboost::obj
