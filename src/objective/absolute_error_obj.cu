/**
 * Copyright 2026, XGBoost Contributors
 * \file absolute_error_obj.cu
 * \brief CUDA implementation of the absolute-error gradient kernel.
 */
#include <dmlc/registry.h>

#include <cmath>   // for fabsf, hypotf, sqrtf
#include <vector>  // for vector

#include "../collective/aggregator.h"   // for GlobalSum
#include "../common/kernel.h"           // for KernelRegistration
#include "../common/linalg_op.cuh"      // for ElementWiseKernel
#include "../common/math.h"             // for CloseTo
#include "../common/numeric.h"          // for Reduce
#include "../common/optional_weight.h"  // for MakeOptionalWeights
#include "../common/transform.h"        // for Transform
#include "absolute_error_obj.h"

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(absolute_error_kernel_cuda);
namespace {
void AbsoluteErrorGradientCuda(Context const* ctx, HostDeviceVector<float> const& preds,
                               MetaInfo const& info, bst_target_t n_targets,
                               linalg::Matrix<GradientPair>* out_gpair) {
  auto device = ctx->Device();
  CHECK(device.IsCUDA());
  auto labels = info.labels.View(device);
  preds.SetDevice(device);
  auto predt = linalg::MakeTensorView(ctx, &preds, info.num_row_, n_targets);
  auto weights = common::MakeOptionalWeights(device, info.weights_);

  HostDeviceVector<float> root_residual(info.num_row_, 0.0f, device);
  std::vector<double> scale_stats(n_targets + 1, 0.0);
  for (bst_target_t target{0}; target < n_targets; ++target) {
    common::Transform<>::Init(
        [=] XGBOOST_DEVICE(std::size_t i, common::Span<float> residual) {
          residual[i] = weights[i] * sqrtf(fabsf(predt(i, target) - labels(i, target)));
        },
        common::Range{0, static_cast<std::int64_t>(info.num_row_)}, ctx->Threads(), device)
        .Eval(&root_residual);
    scale_stats[target] = common::Reduce(ctx, root_residual);
  }
  scale_stats.back() = common::SumOptionalWeights(ctx, weights, info.num_row_);
  auto cpu_ctx = ctx->MakeCPU();
  collective::SafeColl(
      collective::GlobalSum(&cpu_ctx, linalg::MakeVec(scale_stats.data(), scale_stats.size())));

  HostDeviceVector<float> scale(n_targets, 0.0f, device);
  auto scale_h = scale.HostSpan();
  for (bst_target_t target{0}; target < n_targets; ++target) {
    if (!common::CloseTo(scale_stats.back(), 0.0)) {
      auto root_mean = scale_stats[target] / scale_stats.back();
      scale_h[target] = static_cast<float>(root_mean * root_mean);
    }
  }
  scale.SetDevice(device);
  auto scale_d = scale.ConstDeviceSpan();

  out_gpair->SetDevice(device);
  out_gpair->Reshape(info.num_row_, n_targets);
  auto gpair = out_gpair->View(device);
  linalg::cuda_impl::ElementWiseKernel(
      gpair,
      [=] XGBOOST_DEVICE(std::size_t i, std::size_t j) mutable {
        auto residual = predt(i, j) - labels(i, j);
        auto norm = hypotf(scale_d[j], residual);
        auto curvature = norm > 0.0f ? scale_d[j] / norm : 1.0f;
        auto weight = weights[i];
        gpair(i, j) = {weight * residual * curvature, weight * curvature};
      },
      ctx->CUDACtx()->Stream());
}

auto const kRegisterAbsoluteErrorGradientCuda =
    common::KernelRegistration<AbsoluteErrorGradientKernel>{DeviceOrd::kCUDA,
                                                            &AbsoluteErrorGradientCuda};
}  // namespace
}  // namespace xgboost::obj
