/**
 * Copyright 2026, XGBoost Contributors
 * \file tweedie_metric.h
 * \brief Shared declarations for tweedie metrics.
 */
#ifndef XGBOOST_METRIC_TWEEDIE_METRIC_H_
#define XGBOOST_METRIC_TWEEDIE_METRIC_H_

#include <cmath>    // for exp, log
#include <cstdlib>  // for atof
#include <sstream>  // for ostringstream
#include <string>   // for string

#include "elementwise_metric.h"  // for elementwise kernels
#include "xgboost/base.h"        // for bst_float, kRtEps

namespace xgboost::metric {
struct EvalTweedieNLogLik {
  explicit EvalTweedieNLogLik(const char* param) {
    CHECK(param != nullptr) << "tweedie-nloglik must be in format tweedie-nloglik@rho";
    rho_ = atof(param);
    CHECK(rho_ < 2 && rho_ >= 1) << "tweedie variance power must be in interval [1, 2)";
  }
  [[nodiscard]] const char* Name() const {
    static thread_local std::string name;
    std::ostringstream os;
    os << "tweedie-nloglik@" << rho_;
    name = os.str();
    return name.c_str();
  }

  [[nodiscard]] XGBOOST_DEVICE bst_float operator()(bst_float y, bst_float p) const {
    bst_float a = y * std::exp((1 - rho_) * std::log(p)) / (1 - rho_);
    bst_float b = std::exp((2 - rho_) * std::log(p)) / (2 - rho_);
    return -a + b;
  }
  static double GetFinal(double esum, double wsum) { return wsum == 0 ? esum : esum / wsum; }

 protected:
  bst_float rho_;
};

using TweedieNLogLikEvalKernel = elementwise::EvalKernel<EvalTweedieNLogLik>;
}  // namespace xgboost::metric

#endif  // XGBOOST_METRIC_TWEEDIE_METRIC_H_
