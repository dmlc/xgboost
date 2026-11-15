/**
 * Copyright 2026, XGBoost Contributors
 * \file logloss_metric.cc
 * \brief CPU implementation and registration of the log loss metric.
 */
#include "logloss_metric.h"

#include <dmlc/registry.h>

namespace xgboost::metric {
DMLC_REGISTRY_FILE_TAG(logloss_metric);

namespace {
auto const kRegisterLogLossCpu = elementwise::RegisterEvalCpu<EvalRowLogLoss>();
}  // namespace

XGBOOST_REGISTER_METRIC(LogLoss, "logloss")
    .describe("Negative loglikelihood for logistic regression.")
    .set_body([](char const*) {
      return new elementwise::EvalEWiseMetric<EvalRowLogLoss, LogLossEvalKernel>();
    });
}  // namespace xgboost::metric
