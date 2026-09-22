/**
 * Copyright 2019-2026, Contributors
 * \file survival_metric.cc
 * \brief Metrics for survival analysis
 * \author Avinash Barnwal, Hyunsu Cho and Toby Hocking
 */

#include "survival_metric.h"

#include <dmlc/registry.h>

#include <array>
#include <memory>
#include <numeric>  // for accumulate
#include <vector>

#include "../collective/aggregator.h"
#include "../common/kernel.h"
#include "../common/threading_utils.h"
#include "metric_common.h"  // MetricNoCache
#include "xgboost/host_device_vector.h"
#include "xgboost/json.h"
#include "xgboost/metric.h"

namespace xgboost::metric {
DMLC_REGISTRY_FILE_TAG(survival_metric);
namespace {
template <typename Policy>
PackedReduceResult EvalSurvivalCpu(Context const* ctx, HostDeviceVector<float> const& preds,
                                   MetaInfo const& info, Policy policy) {
  auto const& weights = info.weights_;
  auto const& labels_lower_bound = info.labels_lower_bound_;
  auto const& labels_upper_bound = info.labels_upper_bound_;
  auto n_threads = ctx->Threads();
  size_t ndata = labels_lower_bound.Size();
  CHECK_EQ(ndata, labels_upper_bound.Size());

  const auto& h_labels_lower_bound = labels_lower_bound.HostVector();
  const auto& h_labels_upper_bound = labels_upper_bound.HostVector();
  const auto& h_weights = weights.HostVector();
  const auto& h_preds = preds.HostVector();

  std::vector<double> score_tloc(n_threads, 0.0);
  std::vector<double> weight_tloc(n_threads, 0.0);

  common::ParallelFor(ndata, n_threads, [&](size_t i) {
    const double wt = h_weights.empty() ? 1.0 : static_cast<double>(h_weights[i]);
    auto t_idx = omp_get_thread_num();
    score_tloc[t_idx] += policy.EvalRow(static_cast<double>(h_labels_lower_bound[i]),
                                        static_cast<double>(h_labels_upper_bound[i]),
                                        static_cast<double>(h_preds[i])) *
                         wt;
    weight_tloc[t_idx] += wt;
  });

  double residue_sum = std::accumulate(score_tloc.cbegin(), score_tloc.cend(), 0.0);
  double weights_sum = std::accumulate(weight_tloc.cbegin(), weight_tloc.cend(), 0.0);

  PackedReduceResult res{residue_sum, weights_sum};
  return res;
}

template <typename Policy>
auto RegisterSurvivalCpu() {
  return common::KernelRegistration<SurvivalEvalKernel<Policy>>{DeviceOrd::kCPU,
                                                                &EvalSurvivalCpu<Policy>};
}
auto const kRegisterIntervalCpu = RegisterSurvivalCpu<EvalIntervalRegressionAccuracy>();
auto const kRegisterNormalCpu = RegisterSurvivalCpu<EvalAFTNLogLik<common::NormalDistribution>>();
auto const kRegisterLogisticCpu =
    RegisterSurvivalCpu<EvalAFTNLogLik<common::LogisticDistribution>>();
auto const kRegisterExtremeCpu = RegisterSurvivalCpu<EvalAFTNLogLik<common::ExtremeDistribution>>();
}  // namespace

template <typename Policy>
struct EvalEWiseSurvivalBase : public MetricNoCache {
  explicit EvalEWiseSurvivalBase(Context const* ctx) { ctx_ = ctx; }
  EvalEWiseSurvivalBase() = default;

  std::set<std::string> Configure(const Args& args) override {
    auto used = policy_.Configure(args);
    CHECK(ctx_);
    return used;
  }

  double Eval(const HostDeviceVector<float>& preds, const MetaInfo& info) override {
    CheckRowWeights(info);
    CHECK_EQ(preds.Size(), info.labels_lower_bound_.Size());
    CHECK_EQ(preds.Size(), info.labels_upper_bound_.Size());
    CHECK(ctx_);
    auto result = common::DispatchKernel<SurvivalEvalKernel<Policy>>(ctx_, preds, info, policy_);

    std::array<double, 2> dat{result.Residue(), result.Weights()};
    auto rc = collective::GlobalSum(ctx_, linalg::MakeVec(dat.data(), dat.size()));
    collective::SafeColl(rc);
    return Policy::GetFinal(dat[0], dat[1]);
  }

  [[nodiscard]] const char* Name() const override { return policy_.Name(); }

 private:
  Policy policy_;
};

// This class exists because we want to perform dispatch according to the distribution type at
// configuration time, not at prediction time.
struct AFTNLogLikDispatcher : public MetricNoCache {
  [[nodiscard]] const char* Name() const override { return "aft-nloglik"; }

  double Eval(const HostDeviceVector<bst_float>& preds, const MetaInfo& info) override {
    CHECK(metric_) << "AFT metric must be configured first, with distribution type and scale";
    return metric_->Eval(preds, info);
  }

  std::set<std::string> Configure(const Args& args) override {
    auto used = UpdateAndGetUsedParameters(&param_, args);
    switch (param_.aft_loss_distribution) {
      case common::ProbabilityDistributionType::kNormal:
        metric_.reset(new EvalEWiseSurvivalBase<EvalAFTNLogLik<common::NormalDistribution>>(ctx_));
        break;
      case common::ProbabilityDistributionType::kLogistic:
        metric_.reset(
            new EvalEWiseSurvivalBase<EvalAFTNLogLik<common::LogisticDistribution>>(ctx_));
        break;
      case common::ProbabilityDistributionType::kExtreme:
        metric_.reset(new EvalEWiseSurvivalBase<EvalAFTNLogLik<common::ExtremeDistribution>>(ctx_));
        break;
      default:
        LOG(FATAL) << "Unknown probability distribution";
    }
    used.merge(metric_->Configure(args));
    return used;
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String(this->Name());
    out["aft_loss_param"] = ToJson(param_);
  }

  void LoadConfig(const Json& in) override { FromJson(in["aft_loss_param"], &param_); }

 private:
  AFTParam param_;
  std::unique_ptr<MetricNoCache> metric_;
};

XGBOOST_REGISTER_METRIC(AFTNLogLik, "aft-nloglik")
    .describe("Negative log likelihood of Accelerated Failure Time model.")
    .set_body([](const char*) { return new AFTNLogLikDispatcher(); });

XGBOOST_REGISTER_METRIC(IntervalRegressionAccuracy, "interval-regression-accuracy")
    .describe("")
    .set_body([](const char*) {
      return new EvalEWiseSurvivalBase<EvalIntervalRegressionAccuracy>();
    });

}  // namespace xgboost::metric
