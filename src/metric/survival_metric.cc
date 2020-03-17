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
using AFTLoss = xgboost::common::AFTLoss;

namespace xgboost {
namespace metric {
// tag the this file, used by force static link later.
DMLC_REGISTRY_FILE_TAG(survival_metric);

/*! \brief Negative log likelihood of Accelerated Failure Time model */
struct EvalAFT : public Metric {
 public:
  explicit EvalAFT(const char* param) {}

  void Configure(const Args& args) override {
    param_.UpdateAllowUnknown(args);
    loss_.reset(new AFTLoss(param_.aft_loss_distribution));
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String(this->Name());
    out["aft_loss_param"] = toJson(param_);
  }

  void LoadConfig(Json const& in) override {
    fromJson(in["aft_loss_param"], &param_);
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
    double nloglik_sum = 0.0;
    double weight_sum = 0.0;

    const auto& yhat = preds.HostVector();
    const auto& y_lower = info.labels_lower_bound_.HostVector();
    const auto& y_higher = info.labels_upper_bound_.HostVector();
    const auto& weights = info.weights_.HostVector();
    const bool is_null_weight = weights.empty();
    CHECK_LE(yhat.size(), static_cast<size_t>(std::numeric_limits<omp_ulong>::max()))
      << "yhat is too big";
    const omp_ulong nsize = static_cast<omp_ulong>(yhat.size());

    #pragma omp parallel for schedule(static)
    for (omp_ulong i = 0; i < nsize; ++i) {
      // If weights are empty, data is unweighted so we use 1.0 everywhere
      double w = is_null_weight ? 1.0 : weights[i];
      double loss = loss_->Loss(std::log(y_lower[i]), std::log(y_higher[i]),
                                yhat[i], param_.aft_loss_distribution_scale);
      nloglik_sum += loss;
      weight_sum += w;
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
  std::unique_ptr<AFTLoss> loss_;
};

XGBOOST_REGISTER_METRIC(AFT, "aft-nloglik")
.describe("Negative log likelihood of Accelerated Failure Time model.")
.set_body([](const char* param) { return new EvalAFT(param); });

}  // namespace metric
}  // namespace xgboost
