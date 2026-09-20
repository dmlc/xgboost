/**
 * Copyright 2018-2026, XGBoost Contributors
 * \file hinge.cc
 * \brief CPU implementation and registration of the hinge loss objective.
 * \author Henry Gouk
 */
#include "hinge.h"

#include <dmlc/registry.h>

#include <algorithm>  // for max
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t

#include "../common/kernel.h"   // for DispatchKernel
#include "../common/stats.h"    // for SampleMean, WeightedSampleMean
#include "init_estimation.h"    // for CheckInitInputs
#include "xgboost/json.h"       // for Json
#include "xgboost/logging.h"    // for LOG
#include "xgboost/objective.h"  // for ObjFunction

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(hinge_obj);

namespace {
auto const kRegisterHingeGradientCpu = elementwise::RegisterGradientCpu<HingeLoss>();
auto const kRegisterHingePredTransformCpu = elementwise::RegisterTransformCpu<HingeLoss>();
auto const kRegisterHingeValidationCpu = elementwise::RegisterValidationCpu<HingeLabelCheck>();

void CheckHingeLabels(Context const* ctx, MetaInfo const& info) {
  auto valid = common::DispatchKernel<HingeValidationKernel>(ctx, info.labels, HingeLabelCheck{});
  if (!valid) {
    LOG(FATAL) << HingeLoss::LabelErrorMsg();
  }
}
}  // namespace

class HingeObj : public ObjFunction {
 public:
  std::set<std::string> Configure(Args const&) override { return {}; }
  ObjInfo Task() const override { return ObjInfo::kRegression; }

  void InitEstimation(MetaInfo const& info, linalg::Vector<float>* base_score) const override {
    CheckInitInputs(info);
    CheckHingeLabels(ctx_, info);
    if (info.weights_.Empty()) {
      common::SampleMean(ctx_, info.labels, base_score);
    } else {
      common::WeightedSampleMean(ctx_, info.labels, info.weights_, base_score);
    }
    auto intercept = base_score->HostView();
    for (std::size_t i = 0; i < intercept.Size(); ++i) {
      if (intercept(i) > 0.5f) {
        intercept(i) = 1.0f;
      } else if (intercept(i) < 0.5f) {
        intercept(i) = -1.0f;
      } else {
        intercept(i) = 0.0f;
      }
    }
  }

  [[nodiscard]] bst_target_t Targets(MetaInfo const& info) const override {
    // Multi-target regression.
    return std::max(static_cast<std::size_t>(1), info.labels.Shape(1));
  }

  void GetGradient(HostDeviceVector<float> const& preds, MetaInfo const& info, std::int32_t iter,
                   linalg::Matrix<GradientPair>* out_gpair) override {
    CheckInitInputs(info);
    CHECK_EQ(info.labels.Size(), preds.Size()) << "Invalid shape of labels.";
    if (iter == 0) {
      CheckHingeLabels(ctx_, info);
    }

    common::DispatchKernel<HingeGradientKernel>(ctx_, preds, info, this->Targets(info), HingeLoss{},
                                                out_gpair);
  }

  void PredTransform(HostDeviceVector<float>* io_preds) const override {
    common::DispatchKernel<HingePredTransformKernel>(ctx_, io_preds, HingeLoss{});
  }

  [[nodiscard]] const char* DefaultEvalMetric() const override { return "error"; }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("binary:hinge");
  }
  void LoadConfig(Json const&) override {}
};

XGBOOST_REGISTER_OBJECTIVE(HingeObj, "binary:hinge")
    .describe("Hinge loss. Expects labels to be either 0 or 1")
    .set_body([]() { return new HingeObj(); });
}  // namespace xgboost::obj
