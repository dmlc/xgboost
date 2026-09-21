/**
 * Copyright 2026, XGBoost Contributors
 * \file normal_obj.cu
 * \brief CUDA implementations of normal distribution regression kernels.
 */
#include <dmlc/registry.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <thrust/transform_reduce.h>

#include <array>    // for array
#include <cmath>    // for log
#include <cstddef>  // for size_t

#include "../collective/aggregator.h"    // for GlobalSum
#include "../common/cuda_context.cuh"    // for CUDAContext
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
  NormalGradient gradient;
  linalg::cuda_impl::ElementWiseKernel(
      labels,
      [=] XGBOOST_DEVICE(std::size_t i, std::size_t) mutable {
        gradient(predt(i, 0), predt(i, 1), labels(i, 0), weights[i], &gpair(i, 0), &gpair(i, 1));
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

  auto labels = info.labels.View(device);
  auto mean_view = mean.View(device);
  auto weights = common::MakeOptionalWeights(device, info.weights_);
  // Reduce in double before taking log, matching the CPU path without materializing
  // squared residuals in float (which can overflow for finite labels).
  using Stats = thrust::pair<double, double>;
  auto begin = thrust::make_counting_iterator<std::size_t>(0);
  auto result = thrust::transform_reduce(
      ctx->CUDACtx()->CTP(), begin, begin + info.num_row_,
      [=] XGBOOST_DEVICE(std::size_t i) {
        auto diff = static_cast<double>(labels(i, 0)) - mean_view(0);
        auto weight = static_cast<double>(weights[i]);
        return Stats{weight * diff * diff, weight};
      },
      Stats{0.0, 0.0},
      [] XGBOOST_DEVICE(Stats a, Stats b) {
        return Stats{a.first + b.first, a.second + b.second};
      });
  std::array<double, 2> stats{result.first, result.second};
  auto cpu_ctx = ctx->MakeCPU();
  collective::SafeColl(
      collective::GlobalSum(&cpu_ctx, linalg::MakeVec(stats.data(), stats.size())));
  CHECK_GT(stats[1], 0.0);
  auto log_variance = static_cast<float>(std::log(stats[0] / stats[1] + kNormalMinVariance));

  base_score->SetDevice(device);
  base_score->Reshape(2);
  auto out = base_score->View(device);
  dh::LaunchN(1, ctx->CUDACtx()->Stream(), [=] XGBOOST_DEVICE(std::size_t) mutable {
    out(0) = mean_view(0);
    out(1) = log_variance;
  });
}

auto const kRegisterNormalGradientCuda =
    common::KernelRegistration<NormalGradientKernel>{DeviceOrd::kCUDA, &NormalGradientCuda};
auto const kRegisterNormalInitCuda = common::KernelRegistration<NormalInitEstimationKernel>{
    DeviceOrd::kCUDA, &NormalInitEstimationCuda};
}  // namespace
}  // namespace xgboost::obj
