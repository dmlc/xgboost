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

#include "../common/kernel.h"           // for DispatchKernel, KernelRegistration
#include "../common/linalg_op.h"        // for ElementWiseKernel
#include "../common/optional_weight.h"  // for OptionalWeights
#include "../common/threading_utils.h"  // for ParallelFor
#include "init_estimation.h"            // for FitIntercept
#include "xgboost/json.h"               // for Json
#include "xgboost/objective.h"          // for ObjFunction

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(hinge_obj);

namespace cpu_impl {
void HingeGradient(Context const* ctx, HostDeviceVector<float> const& preds, MetaInfo const& info,
                   bst_target_t n_targets, linalg::Matrix<GradientPair>* out_gpair) {
  auto device = DeviceOrd::CPU();
  auto predt = linalg::MakeTensorView(device, preds.ConstHostSpan(), info.num_row_, n_targets);
  auto labels = info.labels.HostView();
  common::OptionalWeights weights{info.weights_.ConstHostSpan()};

  out_gpair->SetDevice(device);
  out_gpair->Reshape(info.num_row_, n_targets);
  auto gpair = out_gpair->HostView();

  linalg::cpu_impl::ElementWiseKernel(
      gpair, ctx->Threads(), [=](std::size_t i, std::size_t j) mutable {
        gpair(i, j) = HingeLoss::Gradient(predt(i, j), labels(i, j), weights[i]);
      });
}

void HingePredTransform(Context const* ctx, HostDeviceVector<float>* preds) {
  auto values = preds->HostSpan();
  common::ParallelFor(values.size(), ctx->Threads(),
                      [=](std::size_t i) { values[i] = HingeLoss::PredTransform(values[i]); });
}
}  // namespace cpu_impl

namespace {
XGBOOST_REGISTER_KERNEL(HingeGradientKernel, "hinge-gradient-cpu", common::MatchCPU,
                        &cpu_impl::HingeGradient);
XGBOOST_REGISTER_KERNEL(HingePredTransformKernel, "hinge-pred-transform-cpu", common::MatchCPU,
                        &cpu_impl::HingePredTransform);
}  // namespace

class HingeObj : public FitIntercept {
 public:
  void Configure(Args const&) override {}
  ObjInfo Task() const override { return ObjInfo::kRegression; }

  [[nodiscard]] bst_target_t Targets(MetaInfo const& info) const override {
    // Multi-target regression.
    return std::max(static_cast<std::size_t>(1), info.labels.Shape(1));
  }

  void GetGradient(HostDeviceVector<float> const& preds, MetaInfo const& info,
                   std::int32_t /*iter*/, linalg::Matrix<GradientPair>* out_gpair) override {
    CheckInitInputs(info);
    CHECK_EQ(info.labels.Size(), preds.Size()) << "Invalid shape of labels.";

    common::DispatchKernel<HingeGradientKernel>(ctx_, preds, info, this->Targets(info), out_gpair);
  }

  void PredTransform(HostDeviceVector<float>* io_preds) const override {
    common::DispatchKernel<HingePredTransformKernel>(ctx_, io_preds);
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
