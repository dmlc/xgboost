/**
 * Copyright 2026, XGBoost Contributors
 * \file alpha_metric.h
 * \brief Shared multi-alpha metric kernels and CPU implementations.
 */
#ifndef XGBOOST_METRIC_ALPHA_METRIC_H_
#define XGBOOST_METRIC_ALPHA_METRIC_H_

#include <cstddef>  // for size_t

#include "../common/kernel.h"
#include "../common/numeric.h"  // for TransformReduce
#include "../common/optional_weight.h"
#include "metric_common.h"
#include "xgboost/context.h"
#include "xgboost/data.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/linalg.h"

namespace xgboost::metric::alpha {
template <typename EvalFn>
struct EvalKernel {
  using Signature = PackedReduceResult(Context const*, HostDeviceVector<float> const&,
                                       MetaInfo const&, HostDeviceVector<float> const&, EvalFn);
};

namespace detail {
template <typename EvalFn>
PackedReduceResult EvalCpu(Context const* ctx, HostDeviceVector<float> const& preds,
                           MetaInfo const& info, HostDeviceVector<float> const& alphas,
                           EvalFn eval) {
  auto labels = info.labels.HostView();
  auto alpha = alphas.ConstHostSpan();
  auto predts = linalg::MakeTensorView(DeviceOrd::CPU(), preds.ConstHostSpan(), info.num_row_,
                                       alphas.Size(), labels.Shape(1));
  common::OptionalWeights weights{info.weights_.ConstHostSpan()};
  return common::TransformReduce(
      predts.Size(), ctx->Threads(), PackedReduceResult{}, [&](std::size_t i) {
        auto [row, alpha_idx, target] = linalg::UnravelIndex(i, predts.Shape());
        float weight = weights[row];
        float loss = eval(predts(row, alpha_idx, target), labels(row, target), alpha[alpha_idx]);
        return PackedReduceResult{loss * weight, weight};
      });
}
}  // namespace detail

template <typename EvalFn>
auto RegisterEvalCpu() {
  return common::KernelRegistration<EvalKernel<EvalFn>>{DeviceOrd::kCPU, &detail::EvalCpu<EvalFn>};
}
}  // namespace xgboost::metric::alpha

#endif  // XGBOOST_METRIC_ALPHA_METRIC_H_
