/*!
 * Copyright 2015 by Contributors
 * \file survival_metric.cc
 * \brief Metrics for Survival Analysis
 * \author Avinash Barnwal, Hyunsu Cho and Prof. Toby Hocking
 */

#include <rabit/rabit.h>
#include <xgboost/metric.h>
#include <dmlc/registry.h>
#include <cmath>

#include <vector>

#include "../common/host_device_vector.h"
#include "../common/math.h"
#include "../common/survival_util.h"

using AFTNoiseDistribution = xgboost::common::AFTNoiseDistribution;
using AFTParam = xgboost::common::AFTParam;
using AFTEventType = xgboost::common::AFTEventType;

namespace xgboost {
namespace metric {
// tag the this file, used by force static link later.
DMLC_REGISTRY_FILE_TAG(survival_metric);

/*! \brief Negative log likelihood of Accelerated Failure Time model */
struct EvalAFT : public Metric {
 public:
  explicit EvalAFT(const char* param) {
    CHECK(param != nullptr) << error_msg_;
    name_ = std::string(param);

    // Split name_ using ',' as delimiter
    // Example: "normal,1.0" will be split into "normal" and "1.0"
    auto pos = name_.find(",");
    CHECK_NE(pos, std::string::npos) << error_msg_;
    std::string dist = name_.substr(0, pos);
    std::string sigma = name_.substr(pos + 1, std::string::npos);
    std::vector<std::pair<std::string, std::string>> kwargs
      = { {"aft_noise_distribution", dist}, {"aft_sigma", sigma} };
    param_.Init(kwargs);
  }

  bst_float Eval(const HostDeviceVector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed) override {
		CHECK_NE(info.labels_lower_bound_.Size(), 0U) << "y_lower cannot be empty";
		CHECK_NE(info.labels_upper_bound_.Size(), 0U) << "y_higher cannot be empty";
    CHECK_EQ(preds.Size(), info.labels_.Size())
      << "label size predict size not match";
    double nloglik = 0.0;

    const auto& yhat     = preds.HostVector();
    const auto& y_lower  = info.labels_lower_bound_.HostVector();
    const auto& y_higher = info.labels_upper_bound_.HostVector();
    const size_t nsize = yhat.size();
    AFTEventType event;

    for (size_t i = 0; i < nsize; ++i) {
      if (y_lower[i] == y_higher[i]) {
        event = AFTEventType::kUncensored;
        nloglik += loss_uncensored(y_lower[i], y_higher[i], yhat[i],
                                   param_.aft_sigma, param_.aft_noise_distribution);
      } else if (!std::isinf(y_lower[i]) && !std::isinf(y_higher[i])) {
        event = AFTEventType::kIntervalCensored;
        nloglik += loss_interval(y_lower[i], y_higher[i], yhat[i],
                                 param_.aft_sigma, param_.aft_noise_distribution);
      } else if (std::isinf(y_lower[i])){
        event = AFTEventType::kLeftCensored;
        nloglik += loss_left(y_lower[i], y_higher[i], yhat[i],
                             param_.aft_sigma, param_.aft_noise_distribution);
      } else if (std::isinf(y_higher[i])) {
        event = AFTEventType::kRightCensored;
        nloglik += loss_right(y_lower[i], y_higher[i], yhat[i],
                              param_.aft_sigma, param_.aft_noise_distribution);
      } else {
        LOG(FATAL) << "AFTObj: Could not determine event type: y_lower = " << y_lower[i]
                   << ", y_higher = " << y_higher[i];
      }
    }

		if (distributed) {
			bst_float dat[1];
			dat[0] = static_cast<bst_float>(nloglik);
			rabit::Allreduce<rabit::op::Sum>(dat, 1);
			return dat[0];
		} else {
			return static_cast<bst_float>(nloglik);
		}
  }

  const char* Name() const override {
    return name_.c_str();
  }

 private:
  std::string name_;
  const char* error_msg_ =
    "AFT must be in form aft-nloglik@[noise-distribution],[sigma] "
    "where [noise-distribution] is one of 'normal', 'logistic', or 'weibull'; "
    "and [sigma] is a positive number";
  AFTParam param_;

  double loss_interval(double y_lower,double y_higher,double y_pred,double sigma,AFTNoiseDistribution dist){
    double z_u;
    double z_l;
    double cdf_u;
    double cdf_l;
    double cost;

    z_u   = (std::log(y_higher) - y_pred)/sigma;
    z_l   = (std::log(y_lower) - y_pred)/sigma;

    switch (dist) {
     case AFTNoiseDistribution::kNormal:
      cdf_u = common::aft::pnorm(z_u,0,1);
      cdf_l = common::aft::pnorm(z_l,0,1);
      break;
     case AFTNoiseDistribution::kLogistic:
      cdf_u = common::aft::plogis(z_u,0,1);
      cdf_l = common::aft::plogis(z_l,0,1);
      break;
     case AFTNoiseDistribution::kWeibull:
      LOG(FATAL) << "Not implemented";
      break;
     default:
      LOG(FATAL) << "Unrecognized AFT noise distribution type";
    }
    cost = -std::log(cdf_u - cdf_l);
    return cost;
  }

  double loss_left(double y_lower,double y_higher,double y_pred,double sigma,AFTNoiseDistribution dist){
    double z;
    double cdf;
    double cost;

    z    = (std::log(y_higher)-y_pred)/sigma;
    switch (dist) {
     case AFTNoiseDistribution::kNormal:
      cdf = common::aft::pnorm(z,0,1);
      break;
     case AFTNoiseDistribution::kLogistic:
      cdf = common::aft::plogis(z,0,1);
      break;
     case AFTNoiseDistribution::kWeibull:
      LOG(FATAL) << "Not implemented";
      break;
     default:
      LOG(FATAL) << "Unrecognized AFT noise distribution type";
    }
    cost = -std::log(cdf);
    return cost;
  }

  double loss_right(double y_lower,double y_higher,double y_pred,double sigma,AFTNoiseDistribution dist){
    double z;
    double cdf;
    double cost;
    z    = (std::log(y_lower)-y_pred)/sigma;
    switch (dist) {
     case AFTNoiseDistribution::kNormal:
      cdf = common::aft::pnorm(z,0,1);
      break;
     case AFTNoiseDistribution::kLogistic:
      cdf = common::aft::plogis(z,0,1);
      break;
     case AFTNoiseDistribution::kWeibull:
      LOG(FATAL) << "Not implemented";
      break;
     default:
      LOG(FATAL) << "Unrecognized AFT noise distribution type";
    }
    cost = -std::log(1-cdf);
    return cost;
  }

  double loss_uncensored(double y_lower,double y_higher,double y_pred,double sigma,AFTNoiseDistribution dist){
    double z;
    double pdf;
    double cost;
    z       = (std::log(y_lower)-y_pred)/sigma;
    switch (dist) {
     case AFTNoiseDistribution::kNormal:
      pdf = common::aft::dnorm(z,0,1);
      break;
     case AFTNoiseDistribution::kLogistic:
      pdf = common::aft::dlogis(z,0,1);
      break;
     case AFTNoiseDistribution::kWeibull:
      LOG(FATAL) << "Not implemented";
      break;
     default:
      LOG(FATAL) << "Unrecognized AFT noise distribution type";
    }
    cost = -std::log(pdf/(sigma*y_lower));
    return cost;
  }

};

XGBOOST_REGISTER_METRIC(AFT, "aft-nloglik")
.describe("Negative log likelihood of Accelerated Failure Time model.")
.set_body([](const char* param) { return new EvalAFT(param); });

}  // namespace metric
}  // namespace xgboost
