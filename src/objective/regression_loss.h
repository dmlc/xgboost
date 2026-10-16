/**
 * Copyright 2017-2026, XGBoost contributors
 */
#ifndef XGBOOST_OBJECTIVE_REGRESSION_LOSS_H_
#define XGBOOST_OBJECTIVE_REGRESSION_LOSS_H_

#include "xgboost/base.h"  // for XGBOOST_DEVICE

namespace xgboost::obj {
// Label validation for Poisson regression (labels must be non-negative)
struct PoissonLabel {
  XGBOOST_DEVICE static bool CheckLabel(float x) { return x >= 0.0f; }
  static const char* LabelErrorMsg() {
    return "label must be non-negative for Poisson/Tweedie regression.";
  }
};

// Label validation for Tweedie regression (labels must be non-negative)
using TweedieLabel = PoissonLabel;
}  // namespace xgboost::obj
#endif  // XGBOOST_OBJECTIVE_REGRESSION_LOSS_H_
