/**
 * Copyright 2026, XGBoost Contributors
 * \file pseudohuber_metric.cc
 * \brief CPU implementation and registration of the mean pseudo-Huber error metric.
 */
#include "pseudohuber_metric.h"

#include <dmlc/registry.h>

#include <array>   // for array
#include <set>     // for set
#include <string>  // for string

#include "../collective/aggregator.h"   // for GlobalSum
#include "../common/kernel.h"           // for DispatchKernel
#include "../common/nvtx_utils.h"       // for xgboost_NVTX_FN_RANGE
#include "../common/pseudo_huber.h"     // for PseudoHuberParam
#include "metric_common.h"              // for CheckRowWeights, MetricNoCache
#include "xgboost/collective/result.h"  // for SafeColl
#include "xgboost/json.h"               // for FromJson, Json, String, ToJson
#include "xgboost/metric.h"             // for Metric

namespace xgboost::metric {
DMLC_REGISTRY_FILE_TAG(pseudohuber_metric);

namespace {
auto const kRegisterPseudoHuberCpu = elementwise::RegisterEvalCpu<EvalRowPseudoHuber>();
}  // namespace

class PseudoErrorLoss : public MetricNoCache {
  PseudoHuberParam param_;

 public:
  const char* Name() const override { return "mphe"; }
  std::set<std::string> Configure(Args const& args) override {
    return UpdateAndGetUsedParameters(&param_, args);
  }
  void LoadConfig(Json const& in) override { FromJson(in["pseudo_huber_param"], &param_); }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String(this->Name());
    out["pseudo_huber_param"] = ToJson(param_);
  }

  double Eval(const HostDeviceVector<bst_float>& preds, const MetaInfo& info) override {
    xgboost_NVTX_FN_RANGE();

    CHECK_EQ(info.labels.Shape(0), info.num_row_);
    CheckRowWeights(info);
    float slope = this->param_.huber_slope;
    CHECK_NE(slope, 0.0) << "slope for pseudo huber cannot be 0.";
    auto result =
        common::DispatchKernel<PseudoHuberEvalKernel>(ctx_, preds, info, EvalRowPseudoHuber{slope});
    std::array<double, 2> dat{result.Residue(), result.Weights()};
    auto rc = collective::GlobalSum(ctx_, linalg::MakeVec(dat.data(), dat.size()));
    collective::SafeColl(rc);
    return dat[1] == 0 ? dat[0] : dat[0] / dat[1];
  }
};

XGBOOST_REGISTER_METRIC(PseudoErrorLoss, "mphe")
    .describe("Mean Pseudo-huber error.")
    .set_body([](char const*) { return new PseudoErrorLoss(); });
}  // namespace xgboost::metric
