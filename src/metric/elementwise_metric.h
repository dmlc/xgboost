/**
 * Copyright 2026, XGBoost Contributors
 * \file elementwise_metric.h
 * \brief Typed elementwise metric kernels and CPU implementations.
 */
#ifndef XGBOOST_METRIC_ELEMENTWISE_METRIC_H_
#define XGBOOST_METRIC_ELEMENTWISE_METRIC_H_

#include <array>    // for array
#include <cstddef>  // for size_t
#include <numeric>  // for accumulate
#include <vector>   // for vector

#include "../collective/aggregator.h"    // for GlobalSum
#include "../common/kernel.h"            // for KernelRegistration
#include "../common/optional_weight.h"   // for OptionalWeights
#include "../common/threading_utils.h"   // for ParallelFor1d
#include "metric_common.h"               // for PackedReduceResult
#include "xgboost/collective/result.h"   // for SafeColl
#include "xgboost/context.h"             // for Context, DeviceOrd
#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/linalg.h"              // for UnravelIndex

namespace xgboost::metric::elementwise {
template <typename EvalFn, typename Kernel>
class EvalEWiseMetric : public MetricNoCache {
 public:
  EvalEWiseMetric() = default;
  explicit EvalEWiseMetric(char const* policy_param) : eval_{policy_param} {}

  double Eval(HostDeviceVector<bst_float> const& preds, MetaInfo const& info) override {
    CHECK_EQ(preds.Size(), info.labels.Size())
        << "label and prediction size not match, "
        << "hint: use merror or mlogloss for multi-class classification";
    if (info.labels.Size() != 0) {
      CHECK_NE(info.labels.Shape(1), 0);
    }
    CheckRowWeights(info);

    auto result = common::DispatchKernel<Kernel>(ctx_, preds, info, eval_);
    std::array<double, 2> values{result.Residue(), result.Weights()};
    auto rc = collective::GlobalSum(ctx_, linalg::MakeVec(values.data(), values.size()));
    collective::SafeColl(rc);
    return EvalFn::GetFinal(values[0], values[1]);
  }

  [[nodiscard]] char const* Name() const override { return eval_.Name(); }

 private:
  EvalFn eval_;
};

template <typename EvalFn>
struct EvalKernel {
  using Signature = PackedReduceResult(Context const*, HostDeviceVector<float> const&,
                                       MetaInfo const&, EvalFn);
};

namespace detail {
template <typename EvalFn>
PackedReduceResult EvalCpu(Context const* ctx, HostDeviceVector<float> const& preds,
                           MetaInfo const& info, EvalFn eval) {
  auto labels = info.labels.HostView();
  auto predts = preds.ConstHostSpan();
  common::OptionalWeights weights{info.weights_.ConstHostSpan()};

  auto n_threads = ctx->Threads();
  std::vector<double> score_tloc(n_threads, 0.0);
  std::vector<double> weight_tloc(n_threads, 0.0);
  std::size_t constexpr kBlockSize = 2048;
  common::ParallelFor1d<kBlockSize>(labels.Size(), n_threads, [&](auto&& block) {
    double sum_score = 0.0;
    double sum_weight = 0.0;
    for (std::size_t i = block.begin(), n = block.end(); i < n; ++i) {
      auto [sample_id, target_id] = linalg::UnravelIndex(i, labels.Shape());
      float weight = weights[sample_id];
      float residue = eval(labels(sample_id, target_id), predts[i]) * weight;
      sum_score += residue;
      sum_weight += weight;
    }

    auto t_idx = omp_get_thread_num();
    score_tloc[t_idx] += sum_score;
    weight_tloc[t_idx] += sum_weight;
  });

  auto residue_sum = std::accumulate(score_tloc.cbegin(), score_tloc.cend(), 0.0);
  auto weights_sum = std::accumulate(weight_tloc.cbegin(), weight_tloc.cend(), 0.0);
  return PackedReduceResult{residue_sum, weights_sum};
}
}  // namespace detail

template <typename EvalFn>
auto RegisterEvalCpu() {
  using Kernel = EvalKernel<EvalFn>;
  return common::KernelRegistration<Kernel>{DeviceOrd::kCPU, &detail::EvalCpu<EvalFn>};
}
}  // namespace xgboost::metric::elementwise

#endif  // XGBOOST_METRIC_ELEMENTWISE_METRIC_H_
