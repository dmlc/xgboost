/**
 * Copyright 2025, XGBoost Contributors
 */
#pragma once

#include <xgboost/base.h>    // for GradientPair, bst_idx_t, bst_target_t
#include <xgboost/linalg.h>  // for Matrix
#include <xgboost/logging.h>
#include <xgboost/span.h>    // for Span

#include <cstddef>  // for size_t

namespace xgboost {
/**
 * @brief Exact second order statistics for one row, produced alongside the gradient.
 *
 * Multinomial objectives use the last class as the reference class. That leaves
 * `n_free = n_classes - 1` free logits and removes the constant shift null direction of the
 * softmax Hessian, which is what makes the matrix invertible under L2 regularization.
 *
 * Only the Hessian is transported. The matching gradient is already carried by
 * `GradientContainer::gpair`, whose first `n_free` columns hold exactly the free class
 * gradients, so nothing is duplicated here.
 *
 * Every row holds the packed lower triangle of its symmetric `n_free x n_free` Hessian
 * inside one contiguous matrix, so a row costs no separate allocation. The packed index
 * convention lives in `common/exact_multinomial/packed_stats.h`; this type stays layout
 * oriented and carries no objective specific math.
 *
 * The transport is `float`, matching the precision of the objective's probability
 * calculation. Accumulating rows into histogram bins and solving for a leaf are the steps
 * that need more precision, and they use double.
 *
 * Host only.
 */
struct ExactHessian {
  /** @brief Packed lower triangles, shaped (n_samples, RowSize()). */
  linalg::Matrix<float> data;
  /** @brief Number of free logits, `n_classes - 1`. */
  bst_target_t n_free{0};

  /**
   * @brief Scalars needed for the packed lower triangle of an `n_free x n_free` matrix.
   *
   * This is the single definition of the packed row length. `common::PackedHessianSize` in
   * `common/exact_multinomial/packed_stats.h` forwards to it, so the layout cannot drift
   * between the transport and the packed views that interpret it.
   */
  [[nodiscard]] static constexpr std::size_t PackedSize(std::size_t n_free) {
    return n_free * (n_free + 1) / 2;
  }

  [[nodiscard]] bool Empty() const { return data.Empty(); }
  /** @brief Number of scalars per row, `PackedSize(n_free)`. */
  [[nodiscard]] std::size_t RowSize() const { return data.Shape(1); }
  [[nodiscard]] bst_idx_t NumRows() const { return data.Shape(0); }

  /** @brief Allocate one contiguous buffer covering every row. */
  void Reshape(bst_idx_t n_samples, bst_target_t n_free_in) {
    this->n_free = n_free_in;
    this->data.Reshape(n_samples, PackedSize(n_free_in));
  }

  /** @brief Release the statistics and return to the empty state. */
  void Clear() {
    this->n_free = 0;
    this->data.Reshape(static_cast<bst_idx_t>(0), static_cast<std::size_t>(0));
  }

  /** @brief The whole buffer. Hoist this out of a hot loop instead of calling HostRow. */
  [[nodiscard]] common::Span<float> HostValues() { return this->data.HostView().Values(); }
  [[nodiscard]] common::Span<float const> HostValues() const {
    return this->data.HostView().Values();
  }

  [[nodiscard]] common::Span<float> HostRow(bst_idx_t i) {
    auto row_size = this->RowSize();
    return this->HostValues().subspan(i * row_size, row_size);
  }
  [[nodiscard]] common::Span<float const> HostRow(bst_idx_t i) const {
    auto row_size = this->RowSize();
    return this->HostValues().subspan(i * row_size, row_size);
  }
};

/**
 * @brief Container for gradient produced by objective.
 */
struct GradientContainer {
  /** @brief Gradient used for multi-target tree split and linear model. */
  linalg::Matrix<GradientPair> gpair;
  /** @brief Gradient used for tree leaf value, optional. */
  linalg::Matrix<GradientPair> value_gpair;
  /**
   * @brief Exact packed Hessian, optional.
   *
   * Empty unless an objective was explicitly asked for it through
   * `ObjFunction::GetGradientAndExactHessian`. Ordinary training never populates it.
   */
  ExactHessian exact_hessian;

  [[nodiscard]] bool HasValueGrad() const noexcept { return !value_gpair.Empty(); }
  [[nodiscard]] bool HasExactHessian() const noexcept { return !exact_hessian.Empty(); }

  /**
   * @brief Invalidate the exact Hessian.
   *
   * The container outlives a single boosting round, so a sidecar left over from an earlier
   * round would otherwise describe a gradient that no longer exists. Whoever starts a new
   * gradient computation calls this; the objective cannot, because it is handed the
   * gradient matrix rather than the container and does not own the sidecar's lifetime.
   */
  void ClearExactHessian() { this->exact_hessian.Clear(); }

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
