/**
 * Copyright 2026, XGBoost Contributors
 * \file rmse_metric.h
 * \brief Shared declarations for the RMSE and RMSLE metrics.
 */
#ifndef XGBOOST_METRIC_RMSE_METRIC_H_
#define XGBOOST_METRIC_RMSE_METRIC_H_

#include <cmath>  // for log1p, sqrt

#include "elementwise_metric.h"  // for elementwise kernels
#include "xgboost/base.h"        // for bst_float

namespace xgboost::metric {
struct EvalRowRMSE {
  char const* Name() const { return "rmse"; }

  XGBOOST_DEVICE bst_float operator()(bst_float label, bst_float pred) const {
    bst_float diff = label - pred;
    return diff * diff;
  }
  static double GetFinal(double esum, double wsum) {
    return wsum == 0 ? std::sqrt(esum) : std::sqrt(esum / wsum);
  }
};

struct EvalRowRMSLE {
  char const* Name() const { return "rmsle"; }

  XGBOOST_DEVICE bst_float operator()(bst_float label, bst_float pred) const {
    bst_float diff = std::log1p(label) - std::log1p(pred);
    return diff * diff;
  }
  static double GetFinal(double esum, double wsum) {
    return wsum == 0 ? std::sqrt(esum) : std::sqrt(esum / wsum);
  }
};

using RMSEEvalKernel = elementwise::EvalKernel<EvalRowRMSE>;
using RMSLEEvalKernel = elementwise::EvalKernel<EvalRowRMSLE>;
}  // namespace xgboost::metric

#endif  // XGBOOST_METRIC_RMSE_METRIC_H_
