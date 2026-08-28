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
#include "init_estimation.h"    // for FitIntercept
#include "xgboost/json.h"       // for Json
#include "xgboost/objective.h"  // for ObjFunction

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(hinge_obj);

namespace {
auto const kRegisterHingeGradientCpu = elementwise::RegisterGradientCpu<HingeLoss>();
auto const kRegisterHingePredTransformCpu = elementwise::RegisterTransformCpu<HingeLoss>();
}  // namespace

class HingeObj : public FitIntercept {
 public:
  std::set<std::string> Configure(Args const&) override { return {}; }
  ObjInfo Task() const override { return ObjInfo::kRegression; }

  [[nodiscard]] bst_target_t Targets(MetaInfo const& info) const override {
    // Multi-target regression.
    return std::max(static_cast<std::size_t>(1), info.labels.Shape(1));
  }

  void GetGradient(HostDeviceVector<float> const& preds, MetaInfo const& info,
                   std::int32_t /*iter*/, linalg::Matrix<GradientPair>* out_gpair) override {
    CheckInitInputs(info);
    CHECK_EQ(info.labels.Size(), preds.Size()) << "Invalid shape of labels.";

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
    .describe("Hinge loss. Expects labels to be in [0,1f]")
    .set_body([]() { return new HingeObj(); });
}  // namespace xgboost::obj
