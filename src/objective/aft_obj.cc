/*!
 * Copyright 2015 by Contributors
 * \file rank.cc
 * \brief Definition of aft loss.
 */

#include <dmlc/omp.h>
#include <xgboost/logging.h>
#include <xgboost/objective.h>
#include <vector>
#include <algorithm>
#include <memory>
#include <utility>
#include <cmath>
#include "../common/math.h"
#include "../common/random.h"
#include "../common/survival_util.h"

using AFTParam = xgboost::common::AFTParam;
using AFTLoss = xgboost::common::AFTLoss;

namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(aft_obj);

class AFTObj : public ObjFunction {
 public:
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
    loss_.reset(new AFTLoss(param_.aft_noise_distribution));
  }

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo& info,
                   int iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {
    /* Boilerplate */
    CHECK_EQ(preds.Size(), info.extra_float_info_.at("label_lower_bound").Size());
    CHECK_EQ(preds.Size(), info.extra_float_info_.at("label_upper_bound").Size());

    const auto& yhat = preds.HostVector();
    const auto& y_lower = info.extra_float_info_.at("label_lower_bound").HostVector();
    const auto& y_higher = info.extra_float_info_.at("label_upper_bound").HostVector();
    const auto& weights = info.weights_.HostVector();
    const bool is_null_weight = weights.empty();

    out_gpair->Resize(yhat.size());
    std::vector<GradientPair>& gpair = out_gpair->HostVector();
    size_t nsize = yhat.size();
    double first_order_grad;
    double second_order_grad;

    for (int i = 0; i < nsize; ++i) {
      // If weights are empty, data is unweighted so we use 1.0 everywhere
      double w = is_null_weight ? 1.0 : weights[i];
      first_order_grad = loss_->Gradient(std::log(y_lower[i]), std::log(y_higher[i]),
                                         yhat[i], param_.aft_sigma);
      second_order_grad = loss_->Hessian(std::log(y_lower[i]), std::log(y_higher[i]),
                                         yhat[i], param_.aft_sigma);
      gpair[i] = GradientPair(first_order_grad * w, second_order_grad * w);
    }
  }

  void PredTransform(HostDeviceVector<bst_float> *io_preds) override {
    std::vector<bst_float> &preds = io_preds->HostVector();
    const long ndata = static_cast<long>(preds.size()); // NOLINT(*)
    #pragma omp parallel for schedule(static)
    for (long j = 0; j < ndata; ++j) {  // NOLINT(*)
      preds[j] = std::exp(preds[j]);
    }
  }

  void EvalTransform(HostDeviceVector<bst_float> *io_preds) override {
  }

  bst_float ProbToMargin(bst_float base_score) const override {
    return std::log(base_score);
  }

  const char* DefaultEvalMetric() const override {
    return "aft_obj";
  }

 private:
  AFTParam param_;
  std::unique_ptr<AFTLoss> loss_;
};

// register the objective functions
XGBOOST_REGISTER_OBJECTIVE(AFTObj, "aft:survival")
.describe("AFT loss function")
.set_body([]() { return new AFTObj(); });

}  // namespace obj
}  // namespace xgboost


