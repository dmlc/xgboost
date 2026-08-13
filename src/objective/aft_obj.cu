/**
 * Copyright 2019-2026, XGBoost Contributors
 * \file aft_obj.cu
 * \brief CUDA implementations of AFT objective kernels.
 */
#include <dmlc/registry.h>

#include <cmath>    // for exp, log
#include <cstddef>  // for size_t

#include "../common/cuda_context.cuh"    // for CUDAContext
#include "../common/device_helpers.cuh"  // for LaunchN
#include "../common/kernel.h"            // for KernelRegistration
#include "aft_obj.h"

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(aft_kernel_cuda);

namespace {
template <typename Distribution>
void AFTGradientCudaImpl(Context const* ctx, HostDeviceVector<float> const& preds,
                         MetaInfo const& info, float scale,
                         linalg::Matrix<GradientPair>* out_gpair) {
  auto device = ctx->Device();
  preds.SetDevice(device);
  info.labels_lower_bound_.SetDevice(device);
  info.labels_upper_bound_.SetDevice(device);
  info.weights_.SetDevice(device);
  auto predt = preds.ConstDeviceSpan();
  auto lower = info.labels_lower_bound_.ConstDeviceSpan();
  auto upper = info.labels_upper_bound_.ConstDeviceSpan();
  auto weights = info.weights_.ConstDeviceSpan();
  auto gpair = out_gpair->View(device);
  bool is_null_weight = weights.empty();
  dh::LaunchN(predt.size(), ctx->CUDACtx()->Stream(), [=] XGBOOST_DEVICE(std::size_t i) mutable {
    auto grad = static_cast<float>(
        common::AFTLoss<Distribution>::Gradient(lower[i], upper[i], predt[i], scale));
    auto hess = static_cast<float>(
        common::AFTLoss<Distribution>::Hessian(lower[i], upper[i], predt[i], scale));
    auto weight = is_null_weight ? 1.0f : weights[i];
    gpair(i, 0) = {grad * weight, hess * weight};
  });
}

void AFTGradientCuda(Context const* ctx, HostDeviceVector<float> const& preds, MetaInfo const& info,
                     common::ProbabilityDistributionType distribution, float scale,
                     linalg::Matrix<GradientPair>* out_gpair) {
  auto device = ctx->Device();
  CHECK(device.IsCUDA());
  out_gpair->SetDevice(device);
  out_gpair->Reshape(preds.Size(), 1);
  switch (distribution) {
    case common::ProbabilityDistributionType::kNormal:
      AFTGradientCudaImpl<common::NormalDistribution>(ctx, preds, info, scale, out_gpair);
      break;
    case common::ProbabilityDistributionType::kLogistic:
      AFTGradientCudaImpl<common::LogisticDistribution>(ctx, preds, info, scale, out_gpair);
      break;
    case common::ProbabilityDistributionType::kExtreme:
      AFTGradientCudaImpl<common::ExtremeDistribution>(ctx, preds, info, scale, out_gpair);
      break;
    default:
      LOG(FATAL) << "Unrecognized distribution";
  }
}

void AFTPredTransformCuda(Context const* ctx, HostDeviceVector<float>* predictions) {
  auto device = ctx->Device();
  predictions->SetDevice(device);
  auto values = predictions->DeviceSpan();
  dh::LaunchN(values.size(), ctx->CUDACtx()->Stream(),
              [=] XGBOOST_DEVICE(std::size_t i) { values[i] = expf(values[i]); });
}

void AFTProbToMarginCuda(Context const* ctx, linalg::Vector<float>* base_score) {
  auto values = base_score->View(ctx->Device());
  dh::LaunchN(values.Size(), ctx->CUDACtx()->Stream(),
              [=] XGBOOST_DEVICE(std::size_t i) mutable { values(i) = logf(values(i)); });
}

auto const kRegisterAFTGradientCuda =
    common::KernelRegistration<AFTGradientKernel>{DeviceOrd::kCUDA, &AFTGradientCuda};
auto const kRegisterAFTPredTransformCuda =
    common::KernelRegistration<AFTPredTransformKernel>{DeviceOrd::kCUDA, &AFTPredTransformCuda};
auto const kRegisterAFTProbToMarginCuda =
    common::KernelRegistration<AFTProbToMarginKernel>{DeviceOrd::kCUDA, &AFTProbToMarginCuda};
}  // namespace
}  // namespace xgboost::obj
