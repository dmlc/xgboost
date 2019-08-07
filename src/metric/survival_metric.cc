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
#include <memory>
#include <vector>

#include "../common/host_device_vector.h"
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

    loss_.reset(new AFTLoss(param_.aft_noise_distribution));
  }

  bst_float Eval(const HostDeviceVector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed) override {
		CHECK_NE(info.labels_lower_bound_.Size(), 0U) << "y_lower cannot be empty";
		CHECK_NE(info.labels_upper_bound_.Size(), 0U) << "y_higher cannot be empty";
    CHECK_EQ(preds.Size(), info.labels_lower_bound_.Size());
    CHECK_EQ(preds.Size(), info.labels_upper_bound_.Size());
    
    double nloglik = 0.0;

    const auto& yhat     = preds.HostVector();
    const auto& y_lower  = info.labels_lower_bound_.HostVector();
    const auto& y_higher = info.labels_upper_bound_.HostVector();
    const size_t nsize = yhat.size();

    for (size_t i = 0; i < nsize; ++i) {
      nloglik += loss_->loss(y_lower[i], y_higher[i], yhat[i], param_.aft_sigma);
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
  std::unique_ptr<AFTLoss> loss_;
};

XGBOOST_REGISTER_METRIC(AFT, "aft-nloglik")
.describe("Negative log likelihood of Accelerated Failure Time model.")
.set_body([](const char* param) { return new EvalAFT(param); });

}  // namespace metric
}  // namespace xgboost
