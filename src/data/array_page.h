/**
 * Copyright 2025, XGBoost Contributors
 */
#pragma once

#include "xgboost/base.h"
#include "xgboost/linalg.h"

namespace xgboost::data {
struct ArrayPage {
  linalg::Matrix<GradientPair> gpairs;
  std::vector<bst_idx_t> batch_ptr;
};
}  // namespace xgboost::data
