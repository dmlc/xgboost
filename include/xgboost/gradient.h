/**
 * Copyright 2025-2026, XGBoost Contributors
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

  [[nodiscard]] static bool HasValueGrad(linalg::Matrix<GradientPair> const& value_gpair) {
    // !(shape<1> == 0) to fend off dask empty workers.
    return !(value_gpair.Empty() && value_gpair.Shape(1) == 0);
  }
  [[nodiscard]] bool HasValueGrad() const noexcept { return HasValueGrad(this->value_gpair); }

  [[nodiscard]] linalg::Matrix<GradientPair> const* Grad() const { return &gpair; }
  [[nodiscard]] linalg::Matrix<GradientPair>* Grad() { return &gpair; }
};
}  // namespace xgboost
