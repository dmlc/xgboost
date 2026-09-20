/**
 * Copyright 2026, XGBoost Contributors
 * \file logloss_metric.h
 * \brief Shared declarations for the log loss metric.
 */
#ifndef XGBOOST_METRIC_LOGLOSS_METRIC_H_
#define XGBOOST_METRIC_LOGLOSS_METRIC_H_

#include <algorithm>  // for max
#include <cmath>      // for log

#include "elementwise_metric.h"  // for elementwise kernels
#include "xgboost/base.h"        // for bst_float

namespace xgboost::metric {
struct EvalRowLogLoss {
  char const* Name() const { return "logloss"; }

  XGBOOST_DEVICE bst_float operator()(bst_float y, bst_float py) const {
    auto xlogy = [](float x, float y) {
      float eps = 1e-16;
      return (x - 0.0f == 0.0f) ? 0.0f : (x * std::log(std::max(y, eps)));
    };
    const bst_float pneg = 1.0f - py;
    return xlogy(-y, py) + xlogy(-(1.0f - y), pneg);
  }
  static double GetFinal(double esum, double wsum) { return wsum == 0 ? esum : esum / wsum; }
};

using LogLossEvalKernel = elementwise::EvalKernel<EvalRowLogLoss>;
}  // namespace xgboost::metric

#endif  // XGBOOST_METRIC_LOGLOSS_METRIC_H_
