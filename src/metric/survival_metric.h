/**
 * Copyright 2026, XGBoost Contributors
 * \file survival_metric.h
 * \brief Typed survival metric kernels and evaluation policies.
 */
#ifndef XGBOOST_METRIC_SURVIVAL_METRIC_H_
#define XGBOOST_METRIC_SURVIVAL_METRIC_H_

#include <cmath>
#include <set>
#include <string>

#include "../common/survival_util.h"
#include "metric_common.h"
#include "xgboost/context.h"
#include "xgboost/data.h"
#include "xgboost/host_device_vector.h"

namespace xgboost::metric {
using AFTParam = common::AFTParam;
template <typename Distribution>
using AFTLoss = common::AFTLoss<Distribution>;

struct EvalIntervalRegressionAccuracy {
  std::set<std::string> Configure(const Args&) { return {}; }

  [[nodiscard]] const char* Name() const { return "interval-regression-accuracy"; }

  XGBOOST_DEVICE double EvalRow(double label_lower_bound, double label_upper_bound,
                                double log_pred) const {
    const double pred = exp(log_pred);
    return (pred >= label_lower_bound && pred <= label_upper_bound) ? 1.0 : 0.0;
  }

  static double GetFinal(double esum, double wsum) { return wsum == 0 ? esum : esum / wsum; }
};

/*! \brief Negative log likelihood of Accelerated Failure Time model */
template <typename Distribution>
struct EvalAFTNLogLik {
  std::set<std::string> Configure(const Args& args) {
    return UpdateAndGetUsedParameters(&param_, args);
  }

  [[nodiscard]] const char* Name() const { return "aft-nloglik"; }

  XGBOOST_DEVICE double EvalRow(double label_lower_bound, double label_upper_bound,
                                double pred) const {
    return AFTLoss<Distribution>::Loss(label_lower_bound, label_upper_bound, pred,
                                       param_.aft_loss_distribution_scale);
  }

  static double GetFinal(double esum, double wsum) { return wsum == 0 ? esum : esum / wsum; }

 private:
  AFTParam param_;
};

template <typename Policy>
struct SurvivalEvalKernel {
  using Signature = PackedReduceResult(Context const*, HostDeviceVector<float> const&,
                                       MetaInfo const&, Policy);
};
}  // namespace xgboost::metric
#endif  // XGBOOST_METRIC_SURVIVAL_METRIC_H_
