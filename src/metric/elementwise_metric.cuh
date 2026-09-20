/**
 * Copyright 2026, XGBoost Contributors
 * \file elementwise_metric.cuh
 * \brief CUDA implementations of the typed elementwise metric kernels.
 */
#ifndef XGBOOST_METRIC_ELEMENTWISE_METRIC_CUH_
#define XGBOOST_METRIC_ELEMENTWISE_METRIC_CUH_

#include <thrust/functional.h>                  // for plus
#include <thrust/iterator/counting_iterator.h>  // for counting_iterator
#include <thrust/transform_reduce.h>            // for transform_reduce

#include <cstddef>  // for size_t

#include "../common/cuda_context.cuh"   // for CUDAContext
#include "../common/kernel.h"           // for KernelRegistration
#include "../common/optional_weight.h"  // for MakeOptionalWeights
#include "elementwise_metric.h"

namespace xgboost::metric::elementwise {
namespace detail {
template <typename EvalFn>
PackedReduceResult EvalCuda(Context const* ctx, HostDeviceVector<float> const& preds,
                            MetaInfo const& info, EvalFn eval) {
  auto device = ctx->Device();
  CHECK(device.IsCUDA());

  auto labels = info.labels.View(device);
  preds.SetDevice(device);
  auto predts = preds.ConstDeviceSpan();
  auto weights = common::MakeOptionalWeights(device, info.weights_);

  thrust::counting_iterator<std::size_t> begin{0};
  auto end = begin + labels.Size();
  return thrust::transform_reduce(
      ctx->CUDACtx()->CTP(), begin, end,
      [=] XGBOOST_DEVICE(std::size_t i) {
        auto [sample_id, target_id] = linalg::UnravelIndex(i, labels.Shape());
        float weight = weights[sample_id];
        float residue = eval(labels(sample_id, target_id), predts[i]) * weight;
        return PackedReduceResult{residue, weight};
      },
      PackedReduceResult{}, thrust::plus<PackedReduceResult>());
}
}  // namespace detail

template <typename EvalFn>
auto RegisterEvalCuda() {
  using Kernel = EvalKernel<EvalFn>;
  return common::KernelRegistration<Kernel>{DeviceOrd::kCUDA, &detail::EvalCuda<EvalFn>};
}
}  // namespace xgboost::metric::elementwise

#endif  // XGBOOST_METRIC_ELEMENTWISE_METRIC_CUH_
