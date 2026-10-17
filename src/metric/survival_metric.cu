/**
 * Copyright 2019-2026, XGBoost Contributors
 * \file survival_metric.cu
 * \brief CUDA survival metric kernels.
 */
#include <dmlc/registry.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <cuda/std/functional>

#include "../common/cuda_context.cuh"
#include "../common/kernel.h"
#include "survival_metric.h"

namespace xgboost::metric {
DMLC_REGISTRY_FILE_TAG(survival_metric_cuda);
namespace {
template <typename Policy>
PackedReduceResult EvalSurvivalCuda(Context const* ctx, HostDeviceVector<float> const& preds,
                                    MetaInfo const& info, Policy policy) {
  auto const& weights = info.weights_;
  auto const& labels_lower_bound = info.labels_lower_bound_;
  auto const& labels_upper_bound = info.labels_upper_bound_;
  preds.SetDevice(ctx->Device());
  labels_lower_bound.SetDevice(ctx->Device());
  labels_upper_bound.SetDevice(ctx->Device());
  weights.SetDevice(ctx->Device());
  dh::safe_cuda(cudaSetDevice(ctx->Ordinal()));
  size_t ndata = labels_lower_bound.Size();
  CHECK_EQ(ndata, labels_upper_bound.Size());

  thrust::counting_iterator<size_t> begin(0);
  thrust::counting_iterator<size_t> end = begin + ndata;

  auto s_label_lower_bound = labels_lower_bound.DeviceSpan();
  auto s_label_upper_bound = labels_upper_bound.DeviceSpan();
  auto s_preds = preds.DeviceSpan();
  auto s_weights = weights.DeviceSpan();

  const bool is_null_weight = (weights.Size() == 0);

  auto d_policy = policy;

  PackedReduceResult result = thrust::transform_reduce(
      ctx->CUDACtx()->CTP(), begin, end,
      [=] XGBOOST_DEVICE(size_t idx) {
        double weight = is_null_weight ? 1.0 : static_cast<double>(s_weights[idx]);
        double residue = d_policy.EvalRow(static_cast<double>(s_label_lower_bound[idx]),
                                          static_cast<double>(s_label_upper_bound[idx]),
                                          static_cast<double>(s_preds[idx]));
        residue *= weight;
        return PackedReduceResult{residue, weight};
      },
      PackedReduceResult(), cuda::std::plus<PackedReduceResult>());

  return result;
}

template <typename Policy>
auto RegisterSurvivalCuda() {
  return common::KernelRegistration<SurvivalEvalKernel<Policy>>{DeviceOrd::kCUDA,
                                                                &EvalSurvivalCuda<Policy>};
}
auto const kRegisterIntervalCuda = RegisterSurvivalCuda<EvalIntervalRegressionAccuracy>();
auto const kRegisterNormalCuda = RegisterSurvivalCuda<EvalAFTNLogLik<common::NormalDistribution>>();
auto const kRegisterLogisticCuda =
    RegisterSurvivalCuda<EvalAFTNLogLik<common::LogisticDistribution>>();
auto const kRegisterExtremeCuda =
    RegisterSurvivalCuda<EvalAFTNLogLik<common::ExtremeDistribution>>();
}  // namespace

}  // namespace xgboost::metric
