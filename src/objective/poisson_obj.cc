/**
 * Copyright 2026, XGBoost Contributors
 * \file poisson_obj.cc
 * \brief CPU implementation and registration of the Poisson objective.
 */
#include "poisson_obj.h"

#include <dmlc/registry.h>

#include <algorithm>  // for max
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t

#include "../common/kernel.h"   // for DispatchKernel
#include "init_estimation.h"    // for CheckInitInputs, FitInterceptGlmLike
#include "xgboost/json.h"       // for Json, String
#include "xgboost/logging.h"    // for CHECK, LOG
#include "xgboost/objective.h"  // for ObjFunction

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(poisson_obj);

namespace {
auto const kRegisterPoissonGradientCpu = elementwise::RegisterGradientCpu<PoissonGradient>();
auto const kRegisterPoissonPredTransformCpu =
    elementwise::RegisterTransformCpu<PoissonPredTransform>();
auto const kRegisterPoissonProbToMarginCpu =
    elementwise::RegisterTransformCpu<PoissonProbToMargin>();
auto const kRegisterPoissonValidationCpu = elementwise::RegisterValidationCpu<PoissonLabelCheck>();
}  // namespace

class PoissonRegression : public FitInterceptGlmLike {
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
          common::DispatchKernel<PoissonValidationKernel>(ctx_, info.labels, PoissonLabelCheck{});
      if (!valid) {
        LOG(FATAL) << PoissonLabel::LabelErrorMsg();
      }
      if (!info.weights_.Empty()) {
        CHECK_EQ(info.weights_.Size(), info.num_row_)
            << "Number of weights should be equal to the number of data points.";
      }
    }
    common::DispatchKernel<PoissonGradientKernel>(ctx_, preds, info, this->Targets(info),
                                                  PoissonGradient{}, out_gpair);
  }

  void PredTransform(HostDeviceVector<float>* io_preds) const override {
    common::DispatchKernel<PoissonPredTransformKernel>(ctx_, io_preds, PoissonPredTransform{});
  }
  void ProbToMargin(linalg::Vector<float>* base_score) const override {
    common::DispatchKernel<PoissonProbToMarginKernel>(ctx_, base_score->Data(),
                                                      PoissonProbToMargin{});
  }
  [[nodiscard]] const char* DefaultEvalMetric() const override { return "poisson-nloglik"; }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("count:poisson");
  }
  void LoadConfig(Json const&) override {}
};

XGBOOST_REGISTER_OBJECTIVE(PoissonRegression, "count:poisson")
    .describe("Poisson regression for count data.")
    .set_body([]() { return new PoissonRegression(); });
}  // namespace xgboost::obj
