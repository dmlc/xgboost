/**
 * Copyright 2026, XGBoost Contributors
 * \file pseudohuber_obj.cc
 * \brief CPU implementation and registration of the pseudo-Huber objective.
 */
#include "pseudohuber_obj.h"

#include <dmlc/registry.h>

#include <algorithm>  // for max
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t

#include "../common/kernel.h"           // for DispatchKernel, KernelRegistration
#include "../common/linalg_op.h"        // for ElementWiseKernel
#include "../common/optional_weight.h"  // for OptionalWeights
#include "../common/pseudo_huber.h"     // for PseudoHuberParam
#include "init_estimation.h"            // for CheckInitInputs, FitIntercept
#include "xgboost/json.h"               // for FromJson, Json, Object, String, ToJson
#include "xgboost/objective.h"          // for ObjFunction

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(pseudohuber_obj);

namespace cpu_impl {
void PseudoHuberGradient(Context const* ctx, HostDeviceVector<float> const& preds,
                         MetaInfo const& info, bst_target_t n_targets, float slope,
                         linalg::Matrix<GradientPair>* out_gpair) {
  auto device = DeviceOrd::CPU();
  auto predt = linalg::MakeTensorView(device, preds.ConstHostSpan(), info.num_row_, n_targets);
  auto labels = info.labels.HostView();
  common::OptionalWeights weights{info.weights_.ConstHostSpan()};

  out_gpair->SetDevice(device);
  out_gpair->Reshape(info.num_row_, n_targets);
  auto gpair = out_gpair->HostView();

  linalg::cpu_impl::ElementWiseKernel(
      gpair, ctx->Threads(), [=](std::size_t i, std::size_t j) mutable {
        gpair(i, j) = PseudoHuberLoss::Gradient(predt(i, j), labels(i, j), weights[i], slope);
      });
}
}  // namespace cpu_impl

namespace {
common::KernelRegistration<PseudoHuberGradientKernel> const kRegisterPseudoHuberGradientCpu{
    DeviceOrd::kCPU, &cpu_impl::PseudoHuberGradient};
}  // namespace

class PseudoHuberRegression : public FitIntercept {
  PseudoHuberParam param_;

 public:
  void Configure(Args const& args) override { param_.UpdateAllowUnknown(args); }
  [[nodiscard]] ObjInfo Task() const override { return ObjInfo::kRegression; }
  [[nodiscard]] bst_target_t Targets(MetaInfo const& info) const override {
    return std::max(static_cast<std::size_t>(1), info.labels.Shape(1));
  }

  void GetGradient(HostDeviceVector<float> const& preds, MetaInfo const& info,
                   std::int32_t /*iter*/, linalg::Matrix<GradientPair>* out_gpair) override {
    CheckInitInputs(info);
    CHECK_EQ(info.labels.Size(), preds.Size()) << "Invalid shape of labels.";
    auto slope = param_.huber_slope;
    CHECK_NE(slope, 0.0) << "slope for pseudo huber cannot be 0.";
    common::DispatchKernel<PseudoHuberGradientKernel>(ctx_, preds, info, this->Targets(info), slope,
                                                      out_gpair);
  }

  [[nodiscard]] const char* DefaultEvalMetric() const override { return "mphe"; }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("reg:pseudohubererror");
    out["pseudo_huber_param"] = ToJson(param_);
  }

  void LoadConfig(Json const& in) override {
    auto const& config = get<Object const>(in);
    if (config.find("pseudo_huber_param") == config.cend()) {
      // The parameter is added in 1.6.
      return;
    }
    FromJson(in["pseudo_huber_param"], &param_);
  }

  [[nodiscard]] Json DefaultMetricConfig() const override {
    CHECK(param_.GetInitialised());
    Json config{Object{}};
    config["name"] = String{this->DefaultEvalMetric()};
    config["pseudo_huber_param"] = ToJson(param_);
    return config;
  }
};

XGBOOST_REGISTER_OBJECTIVE(PseudoHuberRegression, "reg:pseudohubererror")
    .describe("Regression Pseudo Huber error.")
    .set_body([]() { return new PseudoHuberRegression(); });
}  // namespace xgboost::obj
