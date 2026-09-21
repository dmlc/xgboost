/**
 * Copyright 2026, XGBoost Contributors
 * \file expectile_metric.h
 * \brief Shared declarations for the expectile metric.
 */
#ifndef XGBOOST_METRIC_EXPECTILE_METRIC_H_
#define XGBOOST_METRIC_EXPECTILE_METRIC_H_

#include "alpha_metric.h"

namespace xgboost::metric {
struct EvalRowExpectile {
  XGBOOST_DEVICE float operator()(float pred, float label, float alpha) const {
    auto diff = pred - label;
    auto weight_scale = diff >= 0.0f ? (1.0f - alpha) : alpha;
    return weight_scale * diff * diff;
  }
};
using ExpectileEvalKernel = alpha::EvalKernel<EvalRowExpectile>;
}  // namespace xgboost::metric

#endif  // XGBOOST_METRIC_EXPECTILE_METRIC_H_
