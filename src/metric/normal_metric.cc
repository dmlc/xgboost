/**
 * Copyright 2026, XGBoost Contributors
 * \file normal_metric.cc
 * \brief CPU implementation and registration of normal negative log-likelihood.
 */
#include "normal_metric.h"

#include <dmlc/registry.h>

#include <array>    // for array
#include <cstddef>  // for size_t
#include <numeric>  // for accumulate
#include <set>      // for set
#include <string>   // for string
#include <vector>   // for vector

#include "../collective/aggregator.h"   // for GlobalSum
#include "../common/kernel.h"           // for DispatchKernel, KernelRegistration
#include "../common/optional_weight.h"  // for OptionalWeights
#include "../common/threading_utils.h"  // for ParallelFor1d
#include "xgboost/collective/result.h"  // for SafeColl
#include "xgboost/linalg.h"             // for UnravelIndex
#include "xgboost/metric.h"             // for Metric

namespace xgboost::metric {
DMLC_REGISTRY_FILE_TAG(normal_metric);

namespace {
PackedReduceResult EvalCpu(Context const* ctx, HostDeviceVector<float> const& preds,
                           MetaInfo const& info) {
  EvalNormalNLogLik eval;
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
      float residue =
          eval(labels(sample_id, target_id), predts[sample_id * 2], predts[sample_id * 2 + 1]) *
          weight;
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

auto const kRegisterNormalCpu =
    common::KernelRegistration<NormalEvalKernel>{DeviceOrd::kCPU, &EvalCpu};
}  // namespace

class NormalNLogLik : public MetricNoCache {
 public:
  std::set<std::string> Configure(Args const&) override { return {}; }

  double Eval(HostDeviceVector<bst_float> const& preds, MetaInfo const& info) override {
    CHECK_EQ(info.labels.Shape(1), 1) << "Normal NLL requires a single response column.";
    CHECK_EQ(preds.Size(), info.num_row_ * 2)
        << "Normal NLL requires two predictions per row: mean and log variance.";

    CheckRowWeights(info);
    auto result = common::DispatchKernel<NormalEvalKernel>(ctx_, preds, info);

    std::array<double, 2> dat{result.Residue(), result.Weights()};
    auto rc = collective::GlobalSum(ctx_, linalg::MakeVec(dat.data(), dat.size()));
    collective::SafeColl(rc);
    return dat[1] == 0.0 ? dat[0] : dat[0] / dat[1];
  }

  [[nodiscard]] const char* Name() const override { return "normal-nloglik"; }
};

XGBOOST_REGISTER_METRIC(NormalNLogLik, "normal-nloglik")
    .describe("Negative log-likelihood for normal distribution regression.")
    .set_body([](char const*) { return new NormalNLogLik(); });
}  // namespace xgboost::metric
