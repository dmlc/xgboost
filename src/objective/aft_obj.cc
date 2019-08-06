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
using AFTEventType = xgboost::common::AFTEventType;

namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(aft_obj);



class AFTObj : public ObjFunction {
 public:
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
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
    AFTEventType event;

    out_gpair->Resize(yhat.size());
    std::vector<GradientPair>& gpair = out_gpair->HostVector();
    size_t nsize = yhat.size();
    double first_order_grad;
    double second_order_grad;

    for (int i = 0; i < nsize; ++i) {
      if (y_lower[i] == y_higher[i]) {
        if(param_.aft_noise_distribution == "normal"){
          AFTNormal aft;
          event = AFTEventType::kUncensored;
          first_order_grad  = aft::grad_uncensored(y_lower[i], y_higher[i], yhat[i],
                                                param_.aft_sigma, event);
          second_order_grad = aft::hessian_uncensored(y_lower[i], y_higher[i], yhat[i],
                                               param_.aft_sigma, event);
          std::cout<<first_order_grad<<" "<<second_order_grad<<std::endl;
          std::cout<<"first_order_grad second_order_grad"<<std::endl;
        }

      } else if (!std::isinf(y_lower[i]) && !std::isinf(y_higher[i])) {
        AFTNormal aft;
        event = AFTEventType::kIntervalCensored;
        first_order_grad  = aft::grad_interval(y_lower[i], y_higher[i], yhat[i],
                                              param_.aft_sigma,event);
        second_order_grad = aft::hessian_interval(y_lower[i], y_higher[i], yhat[i],
                                             param_.aft_sigma, event);
        std::cout<<first_order_grad<<" "<<second_order_grad<<std::endl;
        std::cout<<"first_order_grad second_order_grad"<<std::endl;



      } else if (std::isinf(y_lower[i])){
        event = AFTEventType::kLeftCensored;
        first_order_grad  = aft::grad_interval(y_lower[i], y_higher[i], yhat[i],
                                          param_.aft_sigma, event);
        second_order_grad = aft::hessian_interval(y_lower[i], y_higher[i], yhat[i],
                                         param_.aft_sigma, event);
        std::cout<<first_order_grad<<" "<<second_order_grad<<std::endl;
        std::cout<<"first_order_grad second_order_grad"<<std::endl;

      } else if (std::isinf(y_higher[i])) {
        event = AFTEventType::kRightCensored;
        first_order_grad  = aft::grad_interval(y_lower[i], y_higher[i], yhat[i],
                                           param_.aft_sigma, event);
        second_order_grad = aft::hessian_interval(y_lower[i], y_higher[i], yhat[i],
                                          param_.aft_sigma, event);
        std::cout<<first_order_grad<<" "<<second_order_grad<<std::endl;
        std::cout<<"first_order_grad second_order_grad"<<std::endl;


      } else {
        LOG(FATAL) << "AFTObj: Could not determine event type: y_lower = " << y_lower[i]
                   << ", y_higher = " << y_higher[i];
      }
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
};

// register the objective functions
XGBOOST_REGISTER_OBJECTIVE(AFTObj, "aft:survival")
.describe("AFT loss function")
.set_body([]() { return new AFTObj(); });

}  // namespace obj
}  // namespace xgboost


