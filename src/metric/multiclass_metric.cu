/**
 * Copyright 2026, XGBoost Contributors
 * \file multiclass_metric.cu
 * \brief CUDA implementations of the multiclass metric kernels.
 */
#include <dmlc/registry.h>
#include <thrust/transform_reduce.h>

#include <cuda/std/functional>

#include "../common/cuda_compat.cuh"
#include "../common/cuda_context.cuh"
#include "../common/device_helpers.cuh"
#include "../common/kernel.h"
#include "multiclass_metric.h"

namespace xgboost::metric {
DMLC_REGISTRY_FILE_TAG(multiclass_metric_cuda);
namespace {
template <typename EvalRowPolicy>
PackedReduceResult EvalCuda(Context const* ctx, HostDeviceVector<float> const& preds,
                            MetaInfo const& info, std::size_t n_class,
                            HostDeviceVector<std::int32_t>* label_error) {
  auto const& labels = *info.labels.Data();
  auto const& weights = info.weights_;
  preds.SetDevice(ctx->Device());
  labels.SetDevice(ctx->Device());
  weights.SetDevice(ctx->Device());
  dh::safe_cuda(cudaSetDevice(ctx->Ordinal()));

  size_t n_data = labels.Size();

  dh::counting_iterator<size_t> begin(0);
  dh::counting_iterator<size_t> end = begin + n_data;

  auto s_labels = labels.DeviceSpan();
  auto s_preds = preds.DeviceSpan();
  auto s_weights = weights.DeviceSpan();

  bool const is_null_weight = weights.Size() == 0;
  label_error->HostVector().assign(1, 0);
  label_error->SetDevice(ctx->Device());
  auto s_label_error = label_error->DeviceSpan();

  PackedReduceResult result = thrust::transform_reduce(
      ctx->CUDACtx()->CTP(), begin, end,
      [=] XGBOOST_DEVICE(size_t idx) {
        bst_float weight = is_null_weight ? 1.0f : s_weights[idx];
        bst_float residue = 0;
        auto label = static_cast<int>(s_labels[idx]);
        if (label >= 0 && label < static_cast<int32_t>(n_class)) {
          residue = EvalRowPolicy::EvalRow(label, &s_preds[idx * n_class], n_class) * weight;
        } else {
#if defined(__CUDA_ARCH__)
          atomicExch(s_label_error.data(), label);
#else
          s_label_error[0] = label;
#endif
        }
        return PackedReduceResult{residue, weight};
      },
      PackedReduceResult(), cuda::std::plus<PackedReduceResult>());
  CheckMultiClassLabel(label_error->ConstHostVector()[0], n_class);

  return result;
}

auto const kRegisterErrorCuda = common::KernelRegistration<MultiClassErrorEvalKernel>{
    DeviceOrd::kCUDA, &EvalCuda<EvalMatchError>};
auto const kRegisterLogLossCuda = common::KernelRegistration<MultiClassLogLossEvalKernel>{
    DeviceOrd::kCUDA, &EvalCuda<EvalMultiLogLoss>};
}  // namespace
}  // namespace xgboost::metric
