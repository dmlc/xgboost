/**
 * Copyright 2026, XGBoost Contributors
 * \file error_metric.cc
 * \brief CPU implementation and registration of the classification error metric.
 */
#include "error_metric.h"

#include <dmlc/registry.h>

namespace xgboost::metric {
DMLC_REGISTRY_FILE_TAG(error_metric);

namespace {
auto const kRegisterErrorCpu = elementwise::RegisterEvalCpu<EvalError>();
}  // namespace

XGBOOST_REGISTER_METRIC(Error, "error")
    .describe("Binary classification error.")
    .set_body([](char const* param) {
      return new elementwise::EvalEWiseMetric<EvalError, ErrorEvalKernel>(param);
    });
}  // namespace xgboost::metric
