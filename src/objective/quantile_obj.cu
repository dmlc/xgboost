/**
 * Copyright 2023-2026, XGBoost contributors
 * \file quantile_obj.cu
 * \brief CUDA implementations of quantile objective kernels.
 */
#include <dmlc/registry.h>
#include <thrust/iterator/counting_iterator.h>

#include <cmath>    // for fabsf, fmaxf, sqrtf, tanhf
#include <cstddef>  // for size_t

#include "../collective/aggregator.cuh"  // for GlobalSum
#include "../common/algorithm.cuh"       // for SegmentedSum
#include "../common/device_helpers.cuh"  // for LaunchN, MakeTransformIterator
#include "../common/kernel.h"            // for KernelRegistration
#include "../common/linalg_op.cuh"       // for ElementWiseKernel
#include "../common/optional_weight.h"   // for MakeOptionalWeights
#include "quantile_obj.h"

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(quantile_kernel_cuda);
namespace {
constexpr float kSmoothingScale{0.04f};
constexpr float kMinSurrogateRatio{3.0e-4f};

void QuantileGradientCuda(Context const* ctx, HostDeviceVector<float> const& preds,
                          MetaInfo const& info, bst_target_t n_targets,
                          HostDeviceVector<float> const& alpha,
                          linalg::Matrix<GradientPair>* out_gpair) {
  auto device = ctx->Device();
  CHECK(device.IsCUDA());
  preds.SetDevice(device);
  alpha.SetDevice(device);
  auto predt = linalg::MakeTensorView(ctx, &preds, info.num_row_, n_targets);
  auto labels = info.labels.View(device);
  auto weights = common::MakeOptionalWeights(device, info.weights_);
  auto alpha_d = alpha.ConstDeviceSpan();

  auto n_rows = info.num_row_;
  auto n_stats = static_cast<std::size_t>(n_targets) + 1;
  linalg::Vector<double> scale_stats = linalg::Zeros<double>(ctx, n_stats);
  auto stats = scale_stats.View(device);
  if (n_rows != 0) {
    auto value_it = dh::MakeTransformIterator<double>(
        thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(std::size_t i) {
          auto target = i / n_rows;
          auto row = i % n_rows;
          if (target == n_targets) {
            return static_cast<double>(weights[row]);
          }
          return static_cast<double>(weights[row] *
                                     sqrtf(fabsf(predt(row, target) - labels(row, 0))));
        });
    auto segment_it = dh::MakeTransformIterator<std::size_t>(
        thrust::make_counting_iterator(0ul),
        [=] XGBOOST_DEVICE(std::size_t segment) { return segment * n_rows; });
    common::SegmentedSum(ctx->CUDACtx()->Stream(), value_it, stats.Values().data(), n_stats,
                         segment_it, segment_it + 1);
  }
  collective::SafeColl(collective::GlobalSum(ctx, stats));

  dh::LaunchN(n_targets, ctx->CUDACtx()->Stream(), [=] XGBOOST_DEVICE(std::size_t target) mutable {
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
  linalg::cuda_impl::ElementWiseKernel(
      gpair,
      [=] XGBOOST_DEVICE(std::size_t i, std::size_t j) mutable {
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
      },
      ctx->CUDACtx()->Stream());
}

void QuantileTransformCuda(Context const* ctx, HostDeviceVector<float>* predictions,
                           std::size_t n_alphas) {
  auto device = ctx->Device();
  CHECK(device.IsCUDA());
  predictions->SetDevice(device);
  auto values = predictions->DeviceSpan();
  auto n_rows = values.size() / n_alphas;
  dh::LaunchN(n_rows, ctx->CUDACtx()->Stream(), [=] XGBOOST_DEVICE(std::size_t row) {
    auto predictions = values.subspan(row * n_alphas, n_alphas);
    for (std::size_t i{1}; i < predictions.size(); ++i) {
      auto value = predictions[i];
      auto pos = i;
      while (pos > 0 && predictions[pos - 1] > value) {
        predictions[pos] = predictions[pos - 1];
        --pos;
      }
      predictions[pos] = value;
    }
  });
}

const auto kRegisterQuantileGradientCuda =
    common::KernelRegistration<QuantileGradientKernel>{DeviceOrd::kCUDA, &QuantileGradientCuda};
const auto kRegisterQuantileTransformCuda =
    common::KernelRegistration<QuantileTransformKernel>{DeviceOrd::kCUDA, &QuantileTransformCuda};
}  // namespace
}  // namespace xgboost::obj
