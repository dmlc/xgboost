/**
 * Copyright 2026, XGBoost Contributors
 * \file normal_obj.cu
 * \brief CUDA implementations of normal distribution regression kernels.
 */
#include <dmlc/registry.h>

#include <cmath>    // for logf
#include <cstddef>  // for size_t

#include "../common/device_helpers.cuh"  // for LaunchN
#include "../common/kernel.h"            // for KernelRegistration
#include "../common/linalg_op.cuh"       // for ElementWiseKernel
#include "../common/optional_weight.h"   // for MakeOptionalWeights
#include "../common/stats.h"             // for SampleMean, WeightedSampleMean
#include "normal_obj.h"

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(normal_kernel_cuda);

namespace {
void NormalGradientCuda(Context const* ctx, HostDeviceVector<float> const& preds,
                        MetaInfo const& info, linalg::Matrix<GradientPair>* out_gpair) {
  auto device = ctx->Device();
  CHECK(device.IsCUDA());
  preds.SetDevice(device);
  auto predt = linalg::MakeTensorView(ctx, &preds, info.num_row_, 2);
  auto labels = info.labels.View(device);
  auto weights = common::MakeOptionalWeights(device, info.weights_);

  out_gpair->SetDevice(device);
  out_gpair->Reshape(info.num_row_, 2);
  auto gpair = out_gpair->View(device);
  linalg::cuda_impl::ElementWiseKernel(
      labels,
      [=] XGBOOST_DEVICE(std::size_t i, std::size_t) mutable {
        NormalGradient{}(predt(i, 0), predt(i, 1), labels(i, 0), weights[i], &gpair(i, 0),
                         &gpair(i, 1));
      },
      ctx->CUDACtx()->Stream());
}

void NormalInitEstimationCuda(Context const* ctx, MetaInfo const& info,
                              linalg::Vector<float>* base_score) {
  auto device = ctx->Device();
  CHECK(device.IsCUDA());

  linalg::Vector<float> mean;
  if (info.weights_.Empty()) {
    common::SampleMean(ctx, info.labels, &mean);
  } else {
    common::WeightedSampleMean(ctx, info.labels, info.weights_, &mean);
  }
  CHECK_EQ(mean.Size(), 1);

  linalg::Matrix<float> squared_residual;
  squared_residual.SetDevice(device);
  squared_residual.Reshape(info.num_row_, 1);
  auto residual = squared_residual.View(device);
  auto labels = info.labels.View(device);
  auto mean_view = mean.View(device);
  linalg::cuda_impl::ElementWiseKernel(
      residual,
      [=] XGBOOST_DEVICE(std::size_t i, std::size_t) mutable {
        auto diff = labels(i, 0) - mean_view(0);
        residual(i, 0) = diff * diff;
      },
      ctx->CUDACtx()->Stream());

  linalg::Vector<float> variance;
  if (info.weights_.Empty()) {
    common::SampleMean(ctx, squared_residual, &variance);
  } else {
    common::WeightedSampleMean(ctx, squared_residual, info.weights_, &variance);
  }
  CHECK_EQ(variance.Size(), 1);

  base_score->SetDevice(device);
  base_score->Reshape(2);
  auto out = base_score->View(device);
  auto variance_view = variance.View(device);
  dh::LaunchN(1, ctx->CUDACtx()->Stream(), [=] XGBOOST_DEVICE(std::size_t) mutable {
    out(0) = mean_view(0);
    out(1) = logf(variance_view(0) + kNormalMinVariance);
  });
}

auto const kRegisterNormalGradientCuda =
    common::KernelRegistration<NormalGradientKernel>{DeviceOrd::kCUDA, &NormalGradientCuda};
auto const kRegisterNormalInitCuda = common::KernelRegistration<NormalInitEstimationKernel>{
    DeviceOrd::kCUDA, &NormalInitEstimationCuda};
}  // namespace
}  // namespace xgboost::obj
