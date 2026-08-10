/**
 * Copyright 2026, XGBoost Contributors
 * \file absolute_error_obj.cu
 * \brief CUDA implementation of the absolute-error gradient kernel.
 */
#include <dmlc/registry.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>

#include <cmath>
#include <cstddef>

#include "../collective/aggregator.cuh"
#include "../common/device_helpers.cuh"  // for LaunchN, MakeTransformIterator
#include "../common/kernel.h"            // for KernelRegistration
#include "../common/linalg_op.cuh"       // for ElementWiseKernel
#include "../common/math.h"              // for CloseTo
#include "../common/numeric.h"           // for Reduce
#include "../common/optional_weight.h"   // for MakeOptionalWeights
#include "../common/stats.h"
#include "../tree/fit_stump.h"
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

  auto n_rows = info.num_row_;
  auto n_stats = static_cast<std::size_t>(n_targets) + 1;
  linalg::Vector<double> scale_stats = linalg::Zeros<double>(ctx, n_stats);
  auto key_it = dh::MakeTransformIterator<bst_target_t>(
      thrust::make_counting_iterator(0ul),
      [=] XGBOOST_DEVICE(std::size_t i) { return static_cast<bst_target_t>(i / n_rows); });
  auto value_it = dh::MakeTransformIterator<double>(
      thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(std::size_t i) {
        auto target = i / n_rows;
        auto row = i % n_rows;
        if (target == n_targets) {
          return static_cast<double>(weights[row]);
        }
        return static_cast<double>(weights[row] *
                                   sqrtf(fabsf(predt(row, target) - labels(row, target))));
      });
  auto stats = scale_stats.View(device);
  auto n_values = n_rows * n_stats;
  thrust::reduce_by_key(ctx->CUDACtx()->CTP(), key_it, key_it + n_values, value_it,
                        thrust::make_discard_iterator(), stats.Values().data());
  collective::SafeColl(collective::GlobalSum(ctx, stats));

  HostDeviceVector<float> scale(n_targets, 0.0f, device);
  auto scale_d = scale.DeviceSpan();
  dh::LaunchN(n_targets, ctx->CUDACtx()->Stream(), [=] XGBOOST_DEVICE(std::size_t target) {
    auto sum_weight = stats(n_targets);
    if (sum_weight != 0.0) {
      auto root_mean = stats(target) / sum_weight;
      scale_d[target] = static_cast<float>(root_mean * root_mean);
    }
  });

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

void AbsoluteErrorInitEstimationCuda(Context const* ctx, MetaInfo const& info,
                                     bst_target_t n_targets, linalg::Vector<float>* base_score) {
  auto device = ctx->Device();
  double sum_weight = info.weights_.Empty() ? static_cast<double>(info.num_row_)
                                            : common::Reduce(ctx, info.weights_);
  auto cpu_ctx = ctx->MakeCPU();
  collective::SafeColl(
      collective::GlobalSum(&cpu_ctx, linalg::MakeVec(&sum_weight, std::size_t{1})));
  if (common::CloseTo(sum_weight, 0.0)) {
    LOG(WARNING) << "Sum of weights is close to 0.0, skipping base score estimation.";
    *base_score = linalg::Zeros<float>(ctx, n_targets);
    return;
  }

  linalg::Vector<float> mean;
  if (info.weights_.Empty()) {
    common::SampleMean(ctx, info.labels, &mean);
  } else {
    common::WeightedSampleMean(ctx, info.labels, info.weights_, &mean);
  }
  CHECK_EQ(mean.Size(), n_targets);
  auto mean_d = mean.View(device);

  HostDeviceVector<float> predt(info.labels.Size(), 0.0f, device);
  auto predt_d = linalg::MakeTensorView(ctx, &predt, info.num_row_, n_targets);
  linalg::cuda_impl::ElementWiseKernel(
      predt_d,
      [=] XGBOOST_DEVICE(std::size_t i, std::size_t target) mutable {
        predt_d(i, target) = mean_d(target);
      },
      ctx->CUDACtx()->Stream());

  linalg::Matrix<GradientPair> gpair;
  AbsoluteErrorGradientCuda(ctx, predt, info, n_targets, &gpair);
  tree::FitStump(ctx, gpair, n_targets, base_score);

  auto out = base_score->View(device);
  dh::LaunchN(n_targets, ctx->CUDACtx()->Stream(),
              [=] XGBOOST_DEVICE(std::size_t target) mutable { out(target) += mean_d(target); });
}
auto const kRegisterAbsoluteErrorGradientCuda =
    common::KernelRegistration<AbsoluteErrorGradientKernel>{DeviceOrd::kCUDA,
                                                            &AbsoluteErrorGradientCuda};
auto const kRegisterAbsoluteErrorInitEstimationCuda =
    common::KernelRegistration<AbsoluteErrorInitEstimationKernel>{DeviceOrd::kCUDA,
                                                                  &AbsoluteErrorInitEstimationCuda};
}  // namespace
}  // namespace xgboost::obj
