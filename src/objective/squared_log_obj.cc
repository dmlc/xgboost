/**
 * Copyright 2026, XGBoost Contributors
 * \file squared_log_obj.cc
 * \brief CPU implementation and registration of the squared-log objective.
 */
#include "squared_log_obj.h"

#include <dmlc/registry.h>

#include <algorithm>  // for max
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t

#include "../common/kernel.h"   // for DispatchKernel
#include "init_estimation.h"    // for CheckInitInputs, FitIntercept
#include "xgboost/json.h"       // for Json, String
#include "xgboost/logging.h"    // for CHECK, LOG
#include "xgboost/objective.h"  // for ObjFunction

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(squared_log_obj);

namespace {
auto const kRegisterSquaredLogGradientCpu = elementwise::RegisterGradientCpu<SquaredLogGradient>();
auto const kRegisterSquaredLogValidationCpu =
    elementwise::RegisterValidationCpu<SquaredLogLabelCheck>();
}  // namespace

class SquaredLogErrorRegression : public FitIntercept {
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
      auto valid = common::DispatchKernel<SquaredLogValidationKernel>(ctx_, info.labels,
                                                                      SquaredLogLabelCheck{});
      if (!valid) {
        LOG(FATAL) << SquaredLogError::LabelErrorMsg();
      }
      if (!info.weights_.Empty()) {
        CHECK_EQ(info.weights_.Size(), info.num_row_)
            << "Number of weights should be equal to the number of data points.";
      }
    }
    common::DispatchKernel<SquaredLogGradientKernel>(ctx_, preds, info, this->Targets(info),
                                                     SquaredLogGradient{}, out_gpair);
  }

  [[nodiscard]] const char* DefaultEvalMetric() const override { return "rmsle"; }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String(SquaredLogError::Name());
  }
  void LoadConfig(Json const&) override {}
};

XGBOOST_REGISTER_OBJECTIVE(SquaredLogErrorRegression, SquaredLogError::Name())
    .describe("Root mean squared log error.")
    .set_body([]() { return new SquaredLogErrorRegression(); });
}  // namespace xgboost::obj
