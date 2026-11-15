/**
 * Copyright 2026, XGBoost Contributors
 * \file normal_metric.cu
 * \brief CUDA implementation of the normal negative log-likelihood kernel.
 */
#include <dmlc/registry.h>
#include <thrust/functional.h>                  // for plus
#include <thrust/iterator/counting_iterator.h>  // for counting_iterator
#include <thrust/transform_reduce.h>

#include <cstddef>  // for size_t

#include "../common/cuda_context.cuh"   // for CUDAContext
#include "../common/kernel.h"           // for KernelRegistration
#include "../common/optional_weight.h"  // for MakeOptionalWeights
#include "normal_metric.h"
#include "xgboost/linalg.h"  // for UnravelIndex

namespace xgboost::metric {
DMLC_REGISTRY_FILE_TAG(normal_metric_cuda);
namespace {
PackedReduceResult EvalCuda(Context const* ctx, HostDeviceVector<float> const& preds,
                            MetaInfo const& info) {
  EvalNormalNLogLik eval;
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
        float residue =
            eval(labels(sample_id, target_id), predts[sample_id * 2], predts[sample_id * 2 + 1]) *
            weight;
        return PackedReduceResult{residue, weight};
      },
      PackedReduceResult{}, thrust::plus<PackedReduceResult>());
}

auto const kRegisterNormalCuda =
    common::KernelRegistration<NormalEvalKernel>{DeviceOrd::kCUDA, &EvalCuda};
}  // namespace
}  // namespace xgboost::metric
