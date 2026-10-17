/**
 * Copyright 2026, XGBoost Contributors
 * \file expectile_metric.cc
 * \brief CPU implementation and registration of the expectile metric.
 */
#include "expectile_metric.h"

#include <dmlc/registry.h>

#include <array>
#include <set>
#include <string>

#include "../collective/aggregator.h"
#include "../common/expectile_loss_utils.h"
#include "xgboost/collective/result.h"
#include "xgboost/json.h"
#include "xgboost/metric.h"

namespace xgboost::metric {
DMLC_REGISTRY_FILE_TAG(expectile_metric);
namespace {
auto const kRegisterExpectileCpu = alpha::RegisterEvalCpu<EvalRowExpectile>();
}  // namespace

class ExpectileError : public MetricNoCache {
  HostDeviceVector<float> alpha_;
  common::ExpectileLossParam param_;

 public:
  std::set<std::string> Configure(Args const& args) override {
    auto used = UpdateAndGetUsedParameters(&param_, args);
    param_.Validate();
    alpha_.HostVector() = param_.expectile_alpha.Get();
    return used;
  }

  double Eval(HostDeviceVector<bst_float> const& preds, const MetaInfo& info) override {
    CHECK(!alpha_.Empty());
    CHECK_EQ(info.labels.Shape(0), info.num_row_) << "Invalid shape of labels.";
    CHECK_EQ(preds.Size(), info.labels.Size() * alpha_.Size())
        << "Prediction size must equal label size times the number of alpha values.";
    if (info.num_row_ == 0) {
      // empty DMatrix on distributed env
      std::array<double, 2> dat{0.0, 0.0};
      auto rc = collective::GlobalSum(ctx_, linalg::MakeVec(dat.data(), dat.size()));
      collective::SafeColl(rc);
      CHECK_GT(dat[1], 0);
      return dat[0] / dat[1];
    }

    CheckRowWeights(info);
    CHECK_NE(info.labels.Shape(1), 0);
    auto result =
        common::DispatchKernel<ExpectileEvalKernel>(ctx_, preds, info, alpha_, EvalRowExpectile{});
    std::array<double, 2> dat{result.Residue(), result.Weights()};
    auto rc = collective::GlobalSum(ctx_, linalg::MakeVec(dat.data(), dat.size()));
    collective::SafeColl(rc);
    CHECK_GT(dat[1], 0);
    return dat[0] / dat[1];
  }

  const char* Name() const override { return "expectile"; }
  void LoadConfig(Json const& in) override {
    auto const& obj = get<Object const>(in);
    auto it = obj.find("expectile_loss_param");
    if (it != obj.cend()) {
      FromJson(it->second, &param_);
      auto const& name = get<String const>(in["name"]);
      CHECK_EQ(name, "expectile");
      param_.Validate();
      alpha_.HostVector() = param_.expectile_alpha.Get();
    }
  }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String(this->Name());
    out["expectile_loss_param"] = ToJson(param_);
  }
};

XGBOOST_REGISTER_METRIC(ExpectileError, "expectile")
    .describe("Expectile regression error.")
    .set_body([](const char*) { return new ExpectileError{}; });
}  // namespace xgboost::metric
