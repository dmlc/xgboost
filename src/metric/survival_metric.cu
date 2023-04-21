/**
 * Copyright 2019-2023 by Contributors
 * \file survival_metric.cu
 * \brief Metrics for survival analysis
 * \author Avinash Barnwal, Hyunsu Cho and Toby Hocking
 */

#include <dmlc/registry.h>

#include <array>
#include <memory>
#include <vector>

#include "../collective/communicator-inl.h"
#include "../common/math.h"
#include "../common/survival_util.h"
#include "../common/threading_utils.h"
#include "metric_common.h"  // MetricNoCache
#include "xgboost/host_device_vector.h"
#include "xgboost/json.h"
#include "xgboost/metric.h"

#if defined(XGBOOST_USE_CUDA)
#include <thrust/execution_policy.h>  // thrust::cuda::par
#include "../common/device_helpers.cuh"
#endif  // XGBOOST_USE_CUDA

using AFTParam = xgboost::common::AFTParam;
using ProbabilityDistributionType = xgboost::common::ProbabilityDistributionType;
template <typename Distribution>
using AFTLoss = xgboost::common::AFTLoss<Distribution>;

namespace xgboost {
namespace metric {
// tag the this file, used by force static link later.
DMLC_REGISTRY_FILE_TAG(survival_metric);

template <typename EvalRow>
class ElementWiseSurvivalMetricsReduction {
 public:
  ElementWiseSurvivalMetricsReduction() = default;
  void Configure(EvalRow policy) {
    policy_ = policy;
  }

  PackedReduceResult
  CpuReduceMetrics(const HostDeviceVector<bst_float> &weights,
                   const HostDeviceVector<bst_float> &labels_lower_bound,
                   const HostDeviceVector<bst_float> &labels_upper_bound,
                   const HostDeviceVector<bst_float> &preds,
                   int32_t n_threads) const {
    size_t ndata = labels_lower_bound.Size();
    CHECK_EQ(ndata, labels_upper_bound.Size());

    const auto& h_labels_lower_bound = labels_lower_bound.HostVector();
    const auto& h_labels_upper_bound = labels_upper_bound.HostVector();
    const auto& h_weights = weights.HostVector();
    const auto& h_preds = preds.HostVector();

    std::vector<double> score_tloc(n_threads, 0.0);
    std::vector<double> weight_tloc(n_threads, 0.0);

    common::ParallelFor(ndata, n_threads, [&](size_t i) {
      const double wt =
          h_weights.empty() ? 1.0 : static_cast<double>(h_weights[i]);
      auto t_idx = omp_get_thread_num();
      score_tloc[t_idx] +=
          policy_.EvalRow(static_cast<double>(h_labels_lower_bound[i]),
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

#if defined(XGBOOST_USE_CUDA)

  PackedReduceResult DeviceReduceMetrics(
      const HostDeviceVector<bst_float>& weights,
      const HostDeviceVector<bst_float>& labels_lower_bound,
      const HostDeviceVector<bst_float>& labels_upper_bound,
      const HostDeviceVector<bst_float>& preds) {
    size_t ndata = labels_lower_bound.Size();
    CHECK_EQ(ndata, labels_upper_bound.Size());

    thrust::counting_iterator<size_t> begin(0);
    thrust::counting_iterator<size_t> end = begin + ndata;

    auto s_label_lower_bound = labels_lower_bound.DeviceSpan();
    auto s_label_upper_bound = labels_upper_bound.DeviceSpan();
    auto s_preds = preds.DeviceSpan();
    auto s_weights = weights.DeviceSpan();

    const bool is_null_weight = (weights.Size() == 0);

    auto d_policy = policy_;

    dh::XGBCachingDeviceAllocator<char> alloc;
    PackedReduceResult result = thrust::transform_reduce(
      thrust::cuda::par(alloc),
      begin, end,
      [=] XGBOOST_DEVICE(size_t idx) {
        double weight = is_null_weight ? 1.0 : static_cast<double>(s_weights[idx]);
        double residue = d_policy.EvalRow(
            static_cast<double>(s_label_lower_bound[idx]),
            static_cast<double>(s_label_upper_bound[idx]),
            static_cast<double>(s_preds[idx]));
        residue *= weight;
        return PackedReduceResult{residue, weight};
      },
      PackedReduceResult(),
      thrust::plus<PackedReduceResult>());

    return result;
  }

#endif  // XGBOOST_USE_CUDA

  PackedReduceResult Reduce(
      const Context &ctx,
      const HostDeviceVector<bst_float>& weights,
      const HostDeviceVector<bst_float>& labels_lower_bound,
      const HostDeviceVector<bst_float>& labels_upper_bound,
      const HostDeviceVector<bst_float>& preds) {
    PackedReduceResult result;

    if (ctx.gpu_id < 0) {
      result = CpuReduceMetrics(weights, labels_lower_bound, labels_upper_bound,
                                preds, ctx.Threads());
    }
#if defined(XGBOOST_USE_CUDA)
    else {  // NOLINT
      preds.SetDevice(ctx.gpu_id);
      labels_lower_bound.SetDevice(ctx.gpu_id);
      labels_upper_bound.SetDevice(ctx.gpu_id);
      weights.SetDevice(ctx.gpu_id);

      dh::safe_cuda(cudaSetDevice(ctx.gpu_id));
      result = DeviceReduceMetrics(weights, labels_lower_bound, labels_upper_bound, preds);
    }
#endif  // defined(XGBOOST_USE_CUDA)
    return result;
  }

 private:
  EvalRow policy_;
};

struct EvalIntervalRegressionAccuracy {
  void Configure(const Args&) {}

  const char* Name() const {
    return "interval-regression-accuracy";
  }

  XGBOOST_DEVICE double EvalRow(
      double label_lower_bound, double label_upper_bound, double log_pred) const {
    const double pred = exp(log_pred);
    return (pred >= label_lower_bound && pred <= label_upper_bound) ? 1.0 : 0.0;
  }

  static double GetFinal(double esum, double wsum) {
    return wsum == 0 ? esum : esum / wsum;
  }
};

/*! \brief Negative log likelihood of Accelerated Failure Time model */
template <typename Distribution>
struct EvalAFTNLogLik {
  void Configure(const Args& args) {
    param_.UpdateAllowUnknown(args);
  }

  const char* Name() const {
    return "aft-nloglik";
  }

  XGBOOST_DEVICE double EvalRow(
      double label_lower_bound, double label_upper_bound, double pred) const {
    return AFTLoss<Distribution>::Loss(
        label_lower_bound, label_upper_bound, pred, param_.aft_loss_distribution_scale);
  }

  static double GetFinal(double esum, double wsum) {
    return wsum == 0 ? esum : esum / wsum;
  }
 private:
  AFTParam param_;
};

template <typename Policy>
struct EvalEWiseSurvivalBase : public MetricNoCache {
  explicit EvalEWiseSurvivalBase(Context const* ctx) { ctx_ = ctx; }
  EvalEWiseSurvivalBase() = default;

  void Configure(const Args& args) override {
    policy_.Configure(args);
    reducer_.Configure(policy_);
    CHECK(ctx_);
  }

  double Eval(const HostDeviceVector<float>& preds, const MetaInfo& info) override {
    CHECK_EQ(preds.Size(), info.labels_lower_bound_.Size());
    CHECK_EQ(preds.Size(), info.labels_upper_bound_.Size());
    CHECK(ctx_);
    auto result = reducer_.Reduce(*ctx_, info.weights_, info.labels_lower_bound_,
                                  info.labels_upper_bound_, preds);

    std::array<double, 2> dat{result.Residue(), result.Weights()};
    collective::GlobalSum(info, &dat);
    return Policy::GetFinal(dat[0], dat[1]);
  }

  const char* Name() const override {
    return policy_.Name();
  }

 private:
  Policy policy_;
  ElementWiseSurvivalMetricsReduction<Policy> reducer_;
  int device_{-1};  // used only for GPU metric
};

// This class exists because we want to perform dispatch according to the distribution type at
// configuration time, not at prediction time.
struct AFTNLogLikDispatcher : public MetricNoCache {
  const char* Name() const override {
    return "aft-nloglik";
  }

  double Eval(const HostDeviceVector<bst_float>& preds, const MetaInfo& info) override {
    CHECK(metric_) << "AFT metric must be configured first, with distribution type and scale";
    return metric_->Eval(preds, info);
  }

  void Configure(const Args& args) override {
    param_.UpdateAllowUnknown(args);
    switch (param_.aft_loss_distribution) {
    case common::ProbabilityDistributionType::kNormal:
      metric_.reset(new EvalEWiseSurvivalBase<EvalAFTNLogLik<common::NormalDistribution>>(ctx_));
      break;
    case common::ProbabilityDistributionType::kLogistic:
      metric_.reset(new EvalEWiseSurvivalBase<EvalAFTNLogLik<common::LogisticDistribution>>(ctx_));
      break;
    case common::ProbabilityDistributionType::kExtreme:
      metric_.reset(new EvalEWiseSurvivalBase<EvalAFTNLogLik<common::ExtremeDistribution>>(ctx_));
      break;
    default:
      LOG(FATAL) << "Unknown probability distribution";
    }
    metric_->Configure(args);
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String(this->Name());
    out["aft_loss_param"] = ToJson(param_);
  }

  void LoadConfig(const Json& in) override {
    FromJson(in["aft_loss_param"], &param_);
  }

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

}  // namespace metric
}  // namespace xgboost
