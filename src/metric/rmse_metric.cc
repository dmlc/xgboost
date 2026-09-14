/**
 * Copyright 2026, XGBoost Contributors
 * \file rmse_metric.cc
 * \brief CPU implementation and registration of the RMSE and RMSLE metrics.
 */
#include "rmse_metric.h"

#include <dmlc/registry.h>

#include <array>  // for array

#include "../collective/aggregator.h"   // for GlobalSum
#include "../common/kernel.h"           // for DispatchKernel
#include "metric_common.h"              // for CheckRowWeights, MetricNoCache
#include "xgboost/collective/result.h"  // for SafeColl
#include "xgboost/metric.h"             // for Metric

namespace xgboost::metric {
DMLC_REGISTRY_FILE_TAG(rmse_metric);

namespace {
auto const kRegisterRMSECpu = elementwise::RegisterEvalCpu<EvalRowRMSE>();
auto const kRegisterRMSLECpu = elementwise::RegisterEvalCpu<EvalRowRMSLE>();
}  // namespace

template <typename EvalFn, typename Kernel>
class EvalEWiseMetric : public MetricNoCache {
 public:
  double Eval(HostDeviceVector<bst_float> const& preds, MetaInfo const& info) override {
    CHECK_EQ(preds.Size(), info.labels.Size())
        << "label and prediction size not match, "
        << "hint: use merror or mlogloss for multi-class classification";
    if (info.labels.Size() != 0) {
      CHECK_NE(info.labels.Shape(1), 0);
    }
    CheckRowWeights(info);

    auto result = common::DispatchKernel<Kernel>(ctx_, preds, info, eval_);
    std::array<double, 2> values{result.Residue(), result.Weights()};
    auto rc = collective::GlobalSum(ctx_, linalg::MakeVec(values.data(), values.size()));
    collective::SafeColl(rc);
    return EvalFn::GetFinal(values[0], values[1]);
  }

  [[nodiscard]] char const* Name() const override { return eval_.Name(); }

 private:
  EvalFn eval_;
};

XGBOOST_REGISTER_METRIC(RMSE, "rmse")
    .describe("Rooted mean square error.")
    .set_body([](char const*) { return new EvalEWiseMetric<EvalRowRMSE, RMSEEvalKernel>(); });

XGBOOST_REGISTER_METRIC(RMSLE, "rmsle")
    .describe("Rooted mean square log error.")
    .set_body([](char const*) { return new EvalEWiseMetric<EvalRowRMSLE, RMSLEEvalKernel>(); });
}  // namespace xgboost::metric
