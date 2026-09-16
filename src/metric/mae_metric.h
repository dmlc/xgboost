/**
 * Copyright 2026, XGBoost Contributors
 * \file mae_metric.h
 * \brief Shared declarations for the MAE and MAPE metrics.
 */
#ifndef XGBOOST_METRIC_MAE_METRIC_H_
#define XGBOOST_METRIC_MAE_METRIC_H_

#include <cmath>  // for abs

#include "elementwise_metric.h"  // for elementwise kernels
#include "xgboost/base.h"        // for bst_float

namespace xgboost::metric {
struct EvalRowMAE {
  const char* Name() const { return "mae"; }

  XGBOOST_DEVICE bst_float operator()(bst_float label, bst_float pred) const {
    return std::abs(label - pred);
  }
  static double GetFinal(double esum, double wsum) { return wsum == 0 ? esum : esum / wsum; }
};

struct EvalRowMAPE {
  const char* Name() const { return "mape"; }
  XGBOOST_DEVICE bst_float operator()(bst_float label, bst_float pred) const {
    return std::abs((label - pred) / label);
  }
  static double GetFinal(double esum, double wsum) { return wsum == 0 ? esum : esum / wsum; }
};

using MAEEvalKernel = elementwise::EvalKernel<EvalRowMAE>;
using MAPEEvalKernel = elementwise::EvalKernel<EvalRowMAPE>;
}  // namespace xgboost::metric

#endif  // XGBOOST_METRIC_MAE_METRIC_H_
