/**
 * Copyright 2026, XGBoost Contributors
 * \file tweedie_obj.cc
 * \brief CPU implementation and registration of the Tweedie objective.
 */
#include "tweedie_obj.h"

#include <dmlc/registry.h>

#include <algorithm>  // for max
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t
#include <sstream>    // for ostringstream
#include <string>     // for string

#include "../common/kernel.h"   // for DispatchKernel
#include "init_estimation.h"    // for CheckInitInputs, FitInterceptGlmLike
#include "xgboost/json.h"       // for FromJson, Json, String, ToJson
#include "xgboost/logging.h"    // for CHECK, LOG
#include "xgboost/objective.h"  // for ObjFunction

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(tweedie_obj);

namespace {
auto const kRegisterTweedieGradientCpu = elementwise::RegisterGradientCpu<TweedieGradient>();
auto const kRegisterTweediePredTransformCpu =
    elementwise::RegisterTransformCpu<TweediePredTransform>();
auto const kRegisterTweedieProbToMarginCpu =
    elementwise::RegisterTransformCpu<TweedieProbToMargin>();
auto const kRegisterTweedieValidationCpu = elementwise::RegisterValidationCpu<TweedieLabelCheck>();
}  // namespace

class TweedieRegression : public FitInterceptGlmLike {
 public:
  std::set<std::string> Configure(Args const& args) override {
    auto used = UpdateAndGetUsedParameters(&param_, args);
    std::ostringstream os;
    os << "tweedie-nloglik@" << param_.tweedie_variance_power;
    metric_ = os.str();
    return used;
  }
  [[nodiscard]] ObjInfo Task() const override { return ObjInfo::kRegression; }
  [[nodiscard]] bst_target_t Targets(MetaInfo const& info) const override {
    return std::max(static_cast<std::size_t>(1), info.labels.Shape(1));
  }

  void GetGradient(HostDeviceVector<float> const& preds, MetaInfo const& info, std::int32_t iter,
                   linalg::Matrix<GradientPair>* out_gpair) override {
    CheckInitInputs(info);
    CHECK_EQ(info.labels.Size(), preds.Size()) << "Invalid shape of labels.";
    if (iter == 0) {
      auto valid =
          common::DispatchKernel<TweedieValidationKernel>(ctx_, info.labels, TweedieLabelCheck{});
      if (!valid) {
        LOG(FATAL) << TweedieLabel::LabelErrorMsg();
      }
      if (!info.weights_.Empty()) {
        CHECK_EQ(info.weights_.Size(), info.num_row_)
            << "Number of weights should be equal to the number of data points.";
      }
    }
    common::DispatchKernel<TweedieGradientKernel>(ctx_, preds, info, this->Targets(info),
                                                  TweedieGradient{param_.tweedie_variance_power},
                                                  out_gpair);
  }

  void PredTransform(HostDeviceVector<float>* io_preds) const override {
    common::DispatchKernel<TweediePredTransformKernel>(ctx_, io_preds, TweediePredTransform{});
  }
  void ProbToMargin(linalg::Vector<float>* base_score) const override {
    common::DispatchKernel<TweedieProbToMarginKernel>(ctx_, base_score->Data(),
                                                      TweedieProbToMargin{});
  }
  [[nodiscard]] const char* DefaultEvalMetric() const override { return metric_.c_str(); }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("reg:tweedie");
    out["tweedie_regression_param"] = ToJson(param_);
  }
  void LoadConfig(Json const& in) override { FromJson(in["tweedie_regression_param"], &param_); }

 private:
  std::string metric_;
  TweedieRegressionParam param_;
};

DMLC_REGISTER_PARAMETER(TweedieRegressionParam);
XGBOOST_REGISTER_OBJECTIVE(TweedieRegression, "reg:tweedie")
    .describe("Tweedie regression for insurance data.")
    .set_body([]() { return new TweedieRegression(); });
}  // namespace xgboost::obj
