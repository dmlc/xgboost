/**
 * Copyright 2026, XGBoost Contributors
 * \file gamma_obj.cc
 * \brief CPU implementation and registration of the gamma objective.
 */
#include "gamma_obj.h"

#include <dmlc/registry.h>

#include <algorithm>  // for all_of, max
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t

#include "../common/kernel.h"   // for DispatchKernel
#include "init_estimation.h"    // for CheckInitInputs, FitInterceptGlmLike
#include "xgboost/json.h"       // for Json, Object, String
#include "xgboost/logging.h"    // for CHECK, LOG
#include "xgboost/objective.h"  // for ObjFunction

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(gamma_obj);

namespace {
auto const kRegisterGammaGradientCpu = elementwise::RegisterGradientCpu<GammaGradient>();
auto const kRegisterGammaPredTransformCpu = elementwise::RegisterTransformCpu<GammaPredTransform>();
auto const kRegisterGammaProbToMarginCpu = elementwise::RegisterTransformCpu<GammaProbToMargin>();
auto const kRegisterGammaValidationCpu = elementwise::RegisterValidationCpu<GammaLabelCheck>();
}  // namespace

class GammaRegression : public FitInterceptGlmLike {
 public:
  std::set<std::string> Configure(Args const&) override { return {}; }
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
          common::DispatchKernel<GammaValidationKernel>(ctx_, info.labels, GammaLabelCheck{});
      if (!valid) {
        LOG(FATAL) << "label must be positive for gamma regression.";
      }
      if (!info.weights_.Empty()) {
        CHECK_EQ(info.weights_.Size(), info.num_row_)
            << "Number of weights should be equal to the number of data points.";
      }
    }
    common::DispatchKernel<GammaGradientKernel>(ctx_, preds, info, this->Targets(info),
                                                GammaGradient{}, out_gpair);
  }

  void PredTransform(HostDeviceVector<float>* io_preds) const override {
    common::DispatchKernel<GammaPredTransformKernel>(ctx_, io_preds, GammaPredTransform{});
  }

  void ProbToMargin(linalg::Vector<float>* base_score) const override {
    auto intercept = base_score->HostView();
    auto valid = std::all_of(linalg::cbegin(intercept), linalg::cend(intercept),
                             [](float value) { return value > 0.0f; });
    CHECK(valid) << "`base_score` must be greater than 0 for gamma regression";
    common::DispatchKernel<GammaProbToMarginKernel>(ctx_, base_score->Data(), GammaProbToMargin{});
  }

  [[nodiscard]] const char* DefaultEvalMetric() const override { return "gamma-deviance"; }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String{"reg:gamma"};
  }

  void LoadConfig(Json const&) override {}
};

XGBOOST_REGISTER_OBJECTIVE(GammaRegression, "reg:gamma")
    .describe("Gamma regression using the gamma deviance loss with log link.")
    .set_body([]() { return new GammaRegression(); });
}  // namespace xgboost::obj
