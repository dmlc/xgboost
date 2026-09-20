/**
 * Copyright 2026, XGBoost Contributors
 * \file gamma_metric.h
 * \brief Shared declarations for gamma metrics.
 */
#ifndef XGBOOST_METRIC_GAMMA_METRIC_H_
#define XGBOOST_METRIC_GAMMA_METRIC_H_

#include <algorithm>  // for max
#include <cmath>      // for log

#include "elementwise_metric.h"  // for elementwise kernels
#include "xgboost/base.h"        // for bst_float, kRtEps

namespace xgboost::metric {
/**
 * Gamma deviance
 *
 *   Expected input:
 *   label >= 0
 *   predt >= 0
 */
struct EvalGammaDeviance {
  [[nodiscard]] const char* Name() const { return "gamma-deviance"; }

  [[nodiscard]] XGBOOST_DEVICE bst_float operator()(bst_float label, bst_float predt) const {
    predt += kRtEps;
    label += kRtEps;
    return std::log(predt / label) + label / predt - 1;
  }

  static double GetFinal(double esum, double wsum) {
    if (wsum <= 0) {
      wsum = kRtEps;
    }
    return 2 * esum / wsum;
  }
};

struct EvalGammaNLogLik {
  static const char* Name() { return "gamma-nloglik"; }

  [[nodiscard]] XGBOOST_DEVICE bst_float operator()(bst_float y, bst_float py) const {
    py = std::max(py, 1e-6f);
    // hardcoded dispersion.
    float constexpr kPsi = 1.0;
    bst_float theta = -1. / py;
    bst_float a = kPsi;
    float b = -std::log(-theta);
    // c = 1. / kPsi^2 * std::log(y/kPsi) - std::log(y) - common::LogGamma(1. / kPsi);
    //   = 1.0f        * std::log(y)      - std::log(y) - 0 = 0
    float c = 0;
    // general form for exponential family.
    return -((y * theta - b) / a + c);
  }
  static double GetFinal(double esum, double wsum) { return wsum == 0 ? esum : esum / wsum; }
};

using GammaDevianceEvalKernel = elementwise::EvalKernel<EvalGammaDeviance>;
using GammaNLogLikEvalKernel = elementwise::EvalKernel<EvalGammaNLogLik>;
}  // namespace xgboost::metric

#endif  // XGBOOST_METRIC_GAMMA_METRIC_H_
