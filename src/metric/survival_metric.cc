/*!
 * Copyright 2019 by Contributors
 * \file survival_metric.cc
 * \brief Metrics for survival analysis
 * \author Avinash Barnwal, Hyunsu Cho and Toby Hocking
 */

#include <rabit/rabit.h>
#include <xgboost/metric.h>
#include <xgboost/host_device_vector.h>
#include <dmlc/registry.h>
#include <cmath>
#include <memory>
#include <vector>
#include <limits>

#include "xgboost/json.h"

#include "../common/math.h"
#include "../common/survival_util.h"

using AFTParam = xgboost::common::AFTParam;
using ProbabilityDistributionType = xgboost::common::ProbabilityDistributionType;
template <typename Distribution>
using AFTLoss = xgboost::common::AFTLoss<Distribution>;

namespace xgboost {
namespace metric {
// tag the this file, used by force static link later.
DMLC_REGISTRY_FILE_TAG(survival_metric);

struct EvalIntervalRegressionAccuracy : public Metric {
 public:
  explicit EvalIntervalRegressionAccuracy(const char* param) {}

  bst_float Eval(const HostDeviceVector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed) override {
    CHECK_NE(info.labels_lower_bound_.Size(), 0U)
      << "y_lower cannot be empty";
    CHECK_NE(info.labels_upper_bound_.Size(), 0U)
      << "y_higher cannot be empty";
    CHECK_EQ(preds.Size(), info.labels_lower_bound_.Size());
    CHECK_EQ(preds.Size(), info.labels_upper_bound_.Size());

    const auto& yhat = preds.HostVector();
    const auto& y_lower = info.labels_lower_bound_.HostVector();
    const auto& y_upper = info.labels_upper_bound_.HostVector();
    const auto& weights = info.weights_.HostVector();
    const bool is_null_weight = weights.empty();
    CHECK_LE(yhat.size(), static_cast<size_t>(std::numeric_limits<omp_ulong>::max()))
      << "yhat is too big";
    const omp_ulong nsize = static_cast<omp_ulong>(yhat.size());

    double acc_sum = 0.0;
    double weight_sum = 0.0;
    #pragma omp parallel for \
      firstprivate(nsize, is_null_weight) shared(weights, y_lower, y_upper, yhat) \
      reduction(+:acc_sum, weight_sum)
    for (omp_ulong i = 0; i < nsize; ++i) {
      const double pred = std::exp(yhat[i]);
      const double w = is_null_weight ? 1.0 : weights[i];
      if (pred >= y_lower[i] && pred <= y_upper[i]) {
        acc_sum += 1.0;
      }
      weight_sum += w;  
    }

    double dat[2]{acc_sum, weight_sum};
    if (distributed) {
      rabit::Allreduce<rabit::op::Sum>(dat, 2);
    }
    return static_cast<bst_float>(dat[0] / dat[1]);
  }

  const char* Name() const override {
    return "interval-regression-accuracy";
  }
};

/*! \brief Negative log likelihood of Accelerated Failure Time model */
struct EvalAFT : public Metric {
 public:
  explicit EvalAFT(const char* param) {}

  void Configure(const Args& args) override {
    param_.UpdateAllowUnknown(args);
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String(this->Name());
    out["aft_loss_param"] = ToJson(param_);
  }

  void LoadConfig(Json const& in) override {
    FromJson(in["aft_loss_param"], &param_);
  }

  template <typename Distribution>
  inline void EvalImpl(
      const std::vector<float>& weights, const std::vector<float>& y_lower,
      const std::vector<float>& y_upper, const std::vector<float>& yhat,
      omp_ulong nsize, bool is_null_weight, double aft_loss_distribution_scale,
      double* out_nloglik_sum, double* out_weight_sum) {
    double nloglik_sum = 0.0;
    double weight_sum = 0.0;
    #pragma omp parallel for \
     shared(weights, y_lower, y_upper, yhat) reduction(+:nloglik_sum, weight_sum)
    for (omp_ulong i = 0; i < nsize; ++i) {
      // If weights are empty, data is unweighted so we use 1.0 everywhere
      const double w = is_null_weight ? 1.0 : weights[i];
      const double loss
        = AFTLoss<Distribution>::Loss(y_lower[i], y_upper[i], yhat[i], aft_loss_distribution_scale);
      nloglik_sum += loss;
      weight_sum += w;
    }
    *out_nloglik_sum = nloglik_sum;
    *out_weight_sum = weight_sum;
  }

  bst_float Eval(const HostDeviceVector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed) override {
    CHECK_NE(info.labels_lower_bound_.Size(), 0U)
      << "y_lower cannot be empty";
    CHECK_NE(info.labels_upper_bound_.Size(), 0U)
      << "y_higher cannot be empty";
    CHECK_EQ(preds.Size(), info.labels_lower_bound_.Size());
    CHECK_EQ(preds.Size(), info.labels_upper_bound_.Size());

    /* Compute negative log likelihood for each data point and compute weighted average */
    const auto& yhat = preds.HostVector();
    const auto& y_lower = info.labels_lower_bound_.HostVector();
    const auto& y_upper = info.labels_upper_bound_.HostVector();
    const auto& weights = info.weights_.HostVector();
    const bool is_null_weight = weights.empty();
    const float aft_loss_distribution_scale = param_.aft_loss_distribution_scale;
    CHECK_LE(yhat.size(), static_cast<size_t>(std::numeric_limits<omp_ulong>::max()))
      << "yhat is too big";
    const omp_ulong nsize = static_cast<omp_ulong>(yhat.size());
    double nloglik_sum, weight_sum;
    switch (param_.aft_loss_distribution) {
    case ProbabilityDistributionType::kNormal:
      EvalImpl<common::NormalDistribution>(weights, y_lower, y_upper, yhat, nsize, is_null_weight,
                                           aft_loss_distribution_scale, &nloglik_sum, &weight_sum);
      break;
    case ProbabilityDistributionType::kLogistic:
      EvalImpl<common::LogisticDistribution>(weights, y_lower, y_upper, yhat, nsize, is_null_weight,
                                             aft_loss_distribution_scale, &nloglik_sum,
                                             &weight_sum);
      break;
    case ProbabilityDistributionType::kExtreme:
      EvalImpl<common::ExtremeDistribution>(weights, y_lower, y_upper, yhat, nsize, is_null_weight,
                                            aft_loss_distribution_scale, &nloglik_sum, &weight_sum);
      break;
    default:
      LOG(FATAL) << "Unrecognized probability distribution type";
    }

    double dat[2]{nloglik_sum, weight_sum};
    if (distributed) {
      rabit::Allreduce<rabit::op::Sum>(dat, 2);
    }
    return static_cast<bst_float>(dat[0] / dat[1]);
  }

  const char* Name() const override {
    return "aft-nloglik";
  }

 private:
  AFTParam param_;
};

XGBOOST_REGISTER_METRIC(AFT, "aft-nloglik")
.describe("Negative log likelihood of Accelerated Failure Time model.")
.set_body([](const char* param) { return new EvalAFT(param); });

XGBOOST_REGISTER_METRIC(IntervalRegressionAccuracy, "interval-regression-accuracy")
.describe("")
.set_body([](const char* param) { return new EvalIntervalRegressionAccuracy(param); });

}  // namespace metric
}  // namespace xgboost
