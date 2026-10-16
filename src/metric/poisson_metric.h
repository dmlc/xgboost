/**
 * Copyright 2026, XGBoost Contributors
 * \file poisson_metric.h
 * \brief Shared declarations for the Poisson negative log-likelihood metric.
 */
#ifndef XGBOOST_METRIC_POISSON_METRIC_H_
#define XGBOOST_METRIC_POISSON_METRIC_H_

#include <cmath>  // for log

#include "../common/math.h"      // for LogGamma
#include "elementwise_metric.h"  // for elementwise kernels
#include "xgboost/base.h"        // for bst_float

namespace xgboost::metric {
struct EvalPoissonNegLogLik {
  [[nodiscard]] const char* Name() const { return "poisson-nloglik"; }

  [[nodiscard]] XGBOOST_DEVICE bst_float operator()(bst_float y, bst_float py) const {
    const bst_float eps = 1e-16f;
    if (py < eps) py = eps;
    return common::LogGamma(y + 1.0f) + py - std::log(py) * y;
  }

  static double GetFinal(double esum, double wsum) { return wsum == 0 ? esum : esum / wsum; }
};

using PoissonEvalKernel = elementwise::EvalKernel<EvalPoissonNegLogLik>;
}  // namespace xgboost::metric

#endif  // XGBOOST_METRIC_POISSON_METRIC_H_
