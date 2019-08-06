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
#include <utility>
#include <math.h>
#include "../common/math.h"
#include "../common/random.h"
#include "../common/survival_util.h"

using AFTNoiseDistribution = xgboost::common::AFTNoiseDistribution;
using AFTParam = xgboost::common::AFTParam;
using AFTLoss = xgboost::common::aft::AFTLoss;

namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(aft_obj);



class AFTObj : public ObjFunction {
 public:
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
    loss_ = new AFTLoss(param_.aft_noise_distribution);
  }

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo& info,
                   int iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {
    /* Boilerplate */
    //CHECK_EQ(preds.Size(), info.labels_.Size());
    CHECK_EQ(preds.Size(), info.labels_lower_bound_.Size());
    CHECK_EQ(preds.Size(), info.labels_upper_bound_.Size());

    const auto& yhat     = preds.HostVector();
    const auto& y_lower  = info.labels_lower_bound_.HostVector();
    const auto& y_higher = info.labels_upper_bound_.HostVector();

    out_gpair->Resize(yhat.size());
    std::vector<GradientPair>& gpair = out_gpair->HostVector();
    size_t nsize = yhat.size();
    double first_order_grad;
    double second_order_grad;

    for (int i = 0; i < nsize; ++i) {
      first_order_grad = loss_->gradient(y_lower[i], y_higher[i], yhat[i], param_.aft_sigma);
      second_order_grad = loss_->hessian(y_lower[i], y_higher[i], yhat[i], param_.aft_sigma);
      gpair[i] = GradientPair(first_order_grad, second_order_grad);
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
  //  PredTransform(io_preds);
  }

  bst_float ProbToMargin(bst_float base_score) const override {
    return std::log(base_score);
  }

  const char* DefaultEvalMetric() const override {
    return "aft_obj";
  }

 private:
  AFTParam param_;
  AFTLoss* loss_;
};

// register the objective functions
XGBOOST_REGISTER_OBJECTIVE(AFTObj, "aft:survival")
.describe("AFT loss function")
.set_body([]() { return new AFTObj(); });

}  // namespace obj
}  // namespace xgboost


