/**
 * Copyright 2025, XGBoost Contributors
 */
#pragma once

#include <xgboost/base.h>    // for GradientPair
#include <xgboost/linalg.h>  // for Matrix

namespace xgboost {
/**
 * @brief Container for gradient produced by objective.
 */
struct GradientContainer {
  /** @brief Gradient used for multi-target tree split and linear model. */
  linalg::Matrix<GradientPair> gpair;
  /** @brief Full gradient for tree leaf value, set by gbtree for projected split gradient. */
  linalg::Matrix<GradientPair> value_gpair;

  [[nodiscard]] bool HasValueGrad() const noexcept { return !value_gpair.Empty(); }

  [[nodiscard]] linalg::Matrix<GradientPair> const* Grad() const { return &gpair; }
  [[nodiscard]] linalg::Matrix<GradientPair>* Grad() { return &gpair; }
};
}  // namespace xgboost
