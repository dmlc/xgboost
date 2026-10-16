/**
 * Copyright 2026, XGBoost Contributors
 * \file pseudohuber_metric.h
 * \brief Shared declarations for the mean pseudo-Huber error metric.
 */
#ifndef XGBOOST_METRIC_PSEUDOHUBER_METRIC_H_
#define XGBOOST_METRIC_PSEUDOHUBER_METRIC_H_

#include <cmath>  // for sqrt

#include "../common/math.h"      // for Sqr
#include "elementwise_metric.h"  // for elementwise kernels
#include "xgboost/base.h"        // for bst_float

namespace xgboost::metric {
struct EvalRowPseudoHuber {
  float slope;

  XGBOOST_DEVICE bst_float operator()(bst_float label, bst_float pred) const {
    auto a = label - pred;
    return common::Sqr(slope) * (std::sqrt((1 + common::Sqr(a / slope))) - 1);
  }
};

using PseudoHuberEvalKernel = elementwise::EvalKernel<EvalRowPseudoHuber>;
}  // namespace xgboost::metric

#endif  // XGBOOST_METRIC_PSEUDOHUBER_METRIC_H_
