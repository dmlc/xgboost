/**
 * Copyright 2025, XGBoost Contributors
 */
#pragma once

#include "xgboost/base.h"
#include "xgboost/linalg.h"

namespace xgboost::data {
struct ArrayPage {
  linalg::Matrix<GradientPair> gpairs;
};
}  // namespace xgboost::data
