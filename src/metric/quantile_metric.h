/**
 * Copyright 2026, XGBoost Contributors
 * \file quantile_metric.h
 * \brief Shared declarations for the quantile metric.
 */
#ifndef XGBOOST_METRIC_QUANTILE_METRIC_H_
#define XGBOOST_METRIC_QUANTILE_METRIC_H_

#include "alpha_metric.h"

namespace xgboost::metric {
struct EvalRowQuantile {
  XGBOOST_DEVICE float operator()(float pred, float label, float alpha) const {
    auto d = label - pred;
    float sign = d >= 0.0f;
    return (alpha * sign * d) - (1.0f - alpha) * (1.0f - sign) * d;
  }
};
using QuantileEvalKernel = alpha::EvalKernel<EvalRowQuantile>;
}  // namespace xgboost::metric

#endif  // XGBOOST_METRIC_QUANTILE_METRIC_H_
