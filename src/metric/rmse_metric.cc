/**
 * Copyright 2026, XGBoost Contributors
 * \file rmse_metric.cc
 * \brief CPU implementation and registration of the RMSE and RMSLE metrics.
 */
#include "rmse_metric.h"

#include <dmlc/registry.h>

namespace xgboost::metric {
DMLC_REGISTRY_FILE_TAG(rmse_metric);

namespace {
auto const kRegisterRMSECpu = elementwise::RegisterEvalCpu<EvalRowRMSE>();
auto const kRegisterRMSLECpu = elementwise::RegisterEvalCpu<EvalRowRMSLE>();
}  // namespace

XGBOOST_REGISTER_METRIC(RMSE, "rmse")
    .describe("Rooted mean square error.")
    .set_body([](char const*) {
      return new elementwise::EvalEWiseMetric<EvalRowRMSE, RMSEEvalKernel>();
    });

XGBOOST_REGISTER_METRIC(RMSLE, "rmsle")
    .describe("Rooted mean square log error.")
    .set_body([](char const*) {
      return new elementwise::EvalEWiseMetric<EvalRowRMSLE, RMSLEEvalKernel>();
    });
}  // namespace xgboost::metric
