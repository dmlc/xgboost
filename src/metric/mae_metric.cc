/**
 * Copyright 2026, XGBoost Contributors
 * \file mae_metric.cc
 * \brief CPU implementation and registration of the MAE and MAPE metrics.
 */
#include "mae_metric.h"

#include <dmlc/registry.h>

namespace xgboost::metric {
DMLC_REGISTRY_FILE_TAG(mae_metric);

namespace {
auto const kRegisterMAECpu = elementwise::RegisterEvalCpu<EvalRowMAE>();
auto const kRegisterMAPECpu = elementwise::RegisterEvalCpu<EvalRowMAPE>();
}  // namespace

XGBOOST_REGISTER_METRIC(MAE, "mae").describe("Mean absolute error.").set_body([](char const*) {
  return new elementwise::EvalEWiseMetric<EvalRowMAE, MAEEvalKernel>();
});

XGBOOST_REGISTER_METRIC(MAPE, "mape")
    .describe("Mean absolute percentage error.")
    .set_body([](char const*) {
      return new elementwise::EvalEWiseMetric<EvalRowMAPE, MAPEEvalKernel>();
    });
}  // namespace xgboost::metric
