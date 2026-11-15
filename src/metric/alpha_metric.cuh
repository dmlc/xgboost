/**
 * Copyright 2026, XGBoost Contributors
 * \file alpha_metric.cuh
 * \brief Shared CUDA implementations for multi-alpha metrics.
 */
#ifndef XGBOOST_METRIC_ALPHA_METRIC_CUH_
#define XGBOOST_METRIC_ALPHA_METRIC_CUH_

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <cstddef>

#include "../common/cuda_context.cuh"
#include "alpha_metric.h"

namespace xgboost::metric::alpha {
namespace detail {
template <typename EvalFn>
PackedReduceResult EvalCuda(Context const* ctx, HostDeviceVector<float> const& preds,
                            MetaInfo const& info, HostDeviceVector<float> const& alphas,
                            EvalFn eval) {
  auto device = ctx->Device();
  CHECK(device.IsCUDA());
  auto labels = info.labels.View(device);
  preds.SetDevice(device);
  alphas.SetDevice(device);
  auto alpha = alphas.ConstDeviceSpan();
  auto predts = linalg::MakeTensorView(device, preds.ConstDeviceSpan(),
                                       info.num_row_, alphas.Size(), labels.Shape(1));
  auto weights = common::MakeOptionalWeights(device, info.weights_);
  thrust::counting_iterator<std::size_t> begin{0};
  return thrust::transform_reduce(
      ctx->CUDACtx()->CTP(), begin, begin + predts.Size(),
      [=] XGBOOST_DEVICE(std::size_t i) {
        auto [row, alpha_idx, target] = linalg::UnravelIndex(i, predts.Shape());
        float weight = weights[row];
        float loss = eval(predts(row, alpha_idx, target), labels(row, target), alpha[alpha_idx]);
        return PackedReduceResult{loss * weight, weight};
      },
      PackedReduceResult{}, thrust::plus<PackedReduceResult>());
}
}  // namespace detail

template <typename EvalFn>
auto RegisterEvalCuda() {
  return common::KernelRegistration<EvalKernel<EvalFn>>{DeviceOrd::kCUDA, &detail::EvalCuda<EvalFn>};
}
}  // namespace xgboost::metric::alpha

#endif  // XGBOOST_METRIC_ALPHA_METRIC_CUH_
