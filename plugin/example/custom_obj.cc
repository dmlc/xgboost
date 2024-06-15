/**
 * Copyright 2015-2023, XGBoost Contributors
 * \file custom_metric.cc
 * \brief This is an example to define plugin of xgboost.
 *  This plugin defines the additional metric function.
 */
#include <xgboost/base.h>
#include <xgboost/parameter.h>
#include <xgboost/objective.h>
#include <xgboost/json.h>

namespace xgboost::obj {
// This is a helpful data structure to define parameters
// You do not have to use it.
// see http://dmlc-core.readthedocs.org/en/latest/parameter.html
// for introduction of this module.
struct MyLogisticParam : public XGBoostParameter<MyLogisticParam> {
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
  void Configure(const Args& args) override { param_.UpdateAllowUnknown(args); }

  [[nodiscard]] ObjInfo Task() const override { return ObjInfo::kRegression; }

  void GetGradient(const HostDeviceVector<float>& preds, MetaInfo const& info,
                   std::int32_t /*iter*/, linalg::Matrix<GradientPair>* out_gpair) override {
    out_gpair->Reshape(info.num_row_, 1);
    const std::vector<float>& preds_h = preds.HostVector();
    auto out_gpair_h = out_gpair->HostView();
    auto const labels_h = info.labels.HostView();
    for (size_t i = 0; i < preds_h.size(); ++i) {
      float w = info.GetWeight(i);
      // scale the negative examples!
      if (labels_h(i) == 0.0f) w *= param_.scale_neg_weight;
      // logistic transformation
      float p = 1.0f / (1.0f + std::exp(-preds_h[i]));
      // this is the gradient
      float grad = (p - labels_h(i)) * w;
      // this is the second order gradient
      float hess = p * (1.0f - p) * w;
      out_gpair_h(i) = GradientPair(grad, hess);
    }
  }
  [[nodiscard]] const char* DefaultEvalMetric() const override {
    return "logloss";
  }
  void PredTransform(HostDeviceVector<float> *io_preds) const override {
    // transform margin value to probability.
    std::vector<float> &preds = io_preds->HostVector();
    for (auto& pred : preds) {
      pred = 1.0f / (1.0f + std::exp(-pred));
    }
  }
  [[nodiscard]] float ProbToMargin(float base_score) const override {
    // transform probability to margin value
    return -std::log(1.0f / base_score - 1.0f);
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("mylogistic");
    out["my_logistic_param"] = ToJson(param_);
  }

  void LoadConfig(Json const& in) override {
    FromJson(in["my_logistic_param"], &param_);
  }

 private:
  MyLogisticParam param_;
};

// Finally register the objective function.
// After it succeeds you can try use xgboost with objective=mylogistic
XGBOOST_REGISTER_OBJECTIVE(MyLogistic, "mylogistic")
.describe("User defined logistic regression plugin")
.set_body([]() { return new MyLogistic(); });

}  // namespace xgboost::obj
