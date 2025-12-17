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

  [[nodiscard]] std::int32_t NumBatches() const {
    if (this->batch_ptr.empty()) {
      return 0;
    }
    return static_cast<std::int32_t>(this->batch_ptr.size() - 1);
  }
};
}  // namespace xgboost::data
