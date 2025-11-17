/**
 * Copyright 2025, XGBoost Contributors
 */
#pragma once

#include <xgboost/base.h>    // for GradientPair
#include <xgboost/linalg.h>  // for Matrix
#include <xgboost/logging.h>

#include <cstddef>  // for size_t

namespace xgboost {
/**
 * @brief Container for gradient produced by objective.
 */
struct GradientContainer {
  /** @brief Gradient used for multi-target tree split and linear model. */
  linalg::Matrix<GradientPair> gpair;
  /** @brief Gradient used for tree leaf value, optional. */
  linalg::Matrix<GradientPair> value_gpair;

  [[nodiscard]] bool HasValueGrad() const noexcept { return !value_gpair.Empty(); }

  [[nodiscard]] std::size_t NumSplitTargets() const noexcept { return gpair.Shape(1); }
  [[nodiscard]] std::size_t NumTargets() const noexcept {
    return HasValueGrad() ? value_gpair.Shape(1) : this->gpair.Shape(1);
  }

  linalg::MatrixView<GradientPair const> ValueGrad(Context const* ctx) const {
    if (HasValueGrad()) {
      return this->value_gpair.View(ctx->Device());
    }
    return this->gpair.View(ctx->Device());
  }

  [[nodiscard]] linalg::Matrix<GradientPair> const* Grad() const { return &gpair; }
  [[nodiscard]] linalg::Matrix<GradientPair>* Grad() { return &gpair; }

  [[nodiscard]] linalg::Matrix<GradientPair> const* FullGradOnly() const {
    if (this->HasValueGrad()) {
      LOG(FATAL) << "Reduced gradient is not yet supported.";
    }
    return this->Grad();
  }
  [[nodiscard]] linalg::Matrix<GradientPair>* FullGradOnly() {
    if (this->HasValueGrad()) {
      LOG(FATAL) << "Reduced gradient is not yet supported.";
    }
    return this->Grad();
  }
};
}  // namespace xgboost
