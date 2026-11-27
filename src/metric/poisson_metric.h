/**
 * Copyright 2026, XGBoost Contributors
 * \file poisson_metric.h
 * \brief Shared declarations for the Poisson negative log-likelihood metric.
 */
#ifndef XGBOOST_METRIC_POISSON_METRIC_H_
#define XGBOOST_METRIC_POISSON_METRIC_H_

#include <cmath>  // for log, isfinite

#include "../common/math.h"      // for LogGamma
#include "elementwise_metric.h"  // for elementwise kernels
#include "xgboost/base.h"        // for bst_float

namespace xgboost::metric {
// Poisson negative log-likelihood without the log(y!) term, taking 0 * log(p) as 0.
[[nodiscard]] XGBOOST_DEVICE inline double PoissonNegLogLikNoConst(double y, double p) {
  return y == 0 || !std::isfinite(p) ? p : p - y * std::log(p);
}

struct EvalPoissonNegLogLik {
  [[nodiscard]] const char* Name() const { return "poisson-nloglik"; }

  [[nodiscard]] XGBOOST_DEVICE bst_float operator()(double y, double py) const {
    constexpr double kEps = 1e-16;
    if (py < kEps) py = kEps;
    return common::LogGamma(y + 1.0) + PoissonNegLogLikNoConst(y, py);
  }

  static double GetFinal(double esum, double wsum) { return wsum == 0 ? esum : esum / wsum; }
};

using PoissonEvalKernel = elementwise::EvalKernel<EvalPoissonNegLogLik>;
}  // namespace xgboost::metric

#endif  // XGBOOST_METRIC_POISSON_METRIC_H_
