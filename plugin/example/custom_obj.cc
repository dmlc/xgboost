/*!
 * Copyright 2015 by Contributors
 * \file custom_metric.cc
 * \brief This is an example to define plugin of xgboost.
 *  This plugin defines the additional metric function.
 */
#include <xgboost/base.h>
#include <dmlc/parameter.h>
#include <xgboost/objective.h>

namespace xgboost {
namespace obj {

// This is a helpful data structure to define parameters
// You do not have to use it.
// see http://dmlc-core.readthedocs.org/en/latest/parameter.html
// for introduction of this module.
struct MyLogisticParam : public dmlc::Parameter<MyLogisticParam> {
  float scale_neg_weight;
  // declare parameters
  DMLC_DECLARE_PARAMETER(MyLogisticParam) {
    DMLC_DECLARE_FIELD(scale_neg_weight).set_default(1.0f).set_lower_bound(0.0f)
        .describe("Scale the weight of negative examples by this factor");
  }
};

DMLC_REGISTER_PARAMETER(MyLogisticParam);

// Define a customized logistic regression objective in C++.
// Implement the interface.
class MyLogistic : public ObjFunction {
 public:
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
  }
  void GetGradient(const std::vector<float> &preds,
                   const MetaInfo &info,
                   int iter,
                   std::vector<bst_gpair> *out_gpair) override {
    out_gpair->resize(preds.size());
    for (size_t i = 0; i < preds.size(); ++i) {
      float w = info.GetWeight(i);
      // scale the negative examples!
      if (info.labels[i] == 0.0f) w *= param_.scale_neg_weight;
      // logistic transoformation
      float p = 1.0f / (1.0f + expf(-preds[i]));
      // this is the gradient
      float grad = (p - info.labels[i]) * w;
      // this is the second order gradient
      float hess = p * (1.0f - p) * w;
      out_gpair->at(i) = bst_gpair(grad, hess);
    }
  }
  const char* DefaultEvalMetric() const override {
    return "error";
  }
  void PredTransform(std::vector<float> *io_preds) override {
    // transform margin value to probability.
    std::vector<float> &preds = *io_preds;
    for (size_t i = 0; i < preds.size(); ++i) {
      preds[i] = 1.0f / (1.0f + expf(-preds[i]));
    }
  }
  float ProbToMargin(float base_score) const override {
    // transform probability to margin value
    return -std::log(1.0f / base_score - 1.0f);
  }

 private:
  MyLogisticParam param_;
};

// Finally register the objective function.
// After it succeeds you can try use xgboost with objective=mylogistic
XGBOOST_REGISTER_OBJECTIVE(MyLogistic, "mylogistic")
.describe("User defined logistic regression plugin")
.set_body([]() { return new MyLogistic(); });

}  // namespace obj
}  // namespace xgboost
