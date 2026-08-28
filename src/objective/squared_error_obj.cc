/**
 * Copyright 2026, XGBoost Contributors
 * \file squared_error_obj.cc
 * \brief CPU implementation and registration of the squared-error objective.
 */
#include "squared_error_obj.h"

#include <dmlc/registry.h>

#include <algorithm>  // for max
#include <cmath>      // for abs
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t

#include "../common/kernel.h"   // for DispatchKernel
#include "init_estimation.h"    // for CheckInitInputs, FitIntercept, FitInterceptGlmLike
#include "regression_param.h"   // for RegLossParam
#include "xgboost/json.h"       // for FromJson, Json, Object, String, ToJson
#include "xgboost/logging.h"    // for CHECK, LOG
#include "xgboost/objective.h"  // for ObjFunction

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(squared_error_obj);
DMLC_REGISTER_PARAMETER(RegLossParam);

namespace {
auto const kRegisterSquaredErrorGradientCpu =
    elementwise::RegisterGradientCpu<SquaredErrorGradient>();
}  // namespace

class SquaredErrorRegression : public FitInterceptGlmLike {
 public:
  std::set<std::string> Configure(Args const& args) override {
    return UpdateAndGetUsedParameters(&param_, args);
  }
  [[nodiscard]] ObjInfo Task() const override { return {ObjInfo::kRegression, true}; }
  [[nodiscard]] bst_target_t Targets(MetaInfo const& info) const override {
    return std::max(static_cast<std::size_t>(1), info.labels.Shape(1));
  }

  void GetGradient(HostDeviceVector<float> const& preds, MetaInfo const& info, std::int32_t,
                   linalg::Matrix<GradientPair>* out_gpair) override {
    CheckInitInputs(info);
    CHECK_EQ(info.labels.Size(), preds.Size()) << "Invalid shape of labels.";
    if (!info.weights_.Empty()) {
      CHECK_EQ(info.weights_.Size(), info.num_row_)
          << "Number of weights should be equal to the number of data points.";
    }
    common::DispatchKernel<SquaredErrorGradientKernel>(
        ctx_, preds, info, this->Targets(info), SquaredErrorGradient{param_.scale_pos_weight},
        out_gpair);
  }

  void InitEstimation(MetaInfo const& info, linalg::Vector<float>* base_score) const override {
    if (std::abs(param_.scale_pos_weight - 1.0f) > kRtEps) {
      FitIntercept::InitEstimation(info, base_score);
    } else {
      FitInterceptGlmLike::InitEstimation(info, base_score);
    }
  }

  [[nodiscard]] const char* DefaultEvalMetric() const override { return "rmse"; }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("reg:squarederror");
    out["reg_loss_param"] = ToJson(param_);
  }
  void LoadConfig(Json const& in) override {
    auto obj = get<Object const>(in);
    auto it = obj.find("reg_loss_param");
    if (it != obj.cend()) {
      FromJson(it->second, &param_);
    }
  }

 private:
  RegLossParam param_;
};

XGBOOST_REGISTER_OBJECTIVE(SquaredErrorRegression, "reg:squarederror")
    .describe("Regression with squared error.")
    .set_body([]() { return new SquaredErrorRegression(); });

XGBOOST_REGISTER_OBJECTIVE(LinearRegression, "reg:linear")
    .describe("Regression with squared error.")
    .set_body([]() {
      LOG(WARNING) << "reg:linear is now deprecated in favor of reg:squarederror.";
      return new SquaredErrorRegression();
    });
}  // namespace xgboost::obj
