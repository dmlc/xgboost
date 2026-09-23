/**
 * Copyright 2026, XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_EXACT_MULTINOMIAL_LEAF_SOLVER_H_
#define XGBOOST_COMMON_EXACT_MULTINOMIAL_LEAF_SOLVER_H_

#include <xgboost/logging.h>  // for CHECK_EQ, CHECK_GE
#include <xgboost/span.h>     // for Span

#include <cmath>    // for isfinite
#include <cstddef>  // for size_t
#include <vector>   // for vector

#include "packed_stats.h"  // for PackedMultinomialStats, PackedHessianIndex

namespace xgboost::common {
/**
 * @brief L2 regularization for the exact Newton system, `R`.
 *
 * The solve happens in `K-1` free coordinates while the model emits `K` outputs, so the
 * embedding is a gauge choice. Penalising the *centered* K-output leaf gives
 *
 *   ||delta||^2 = w^T (I - 11^T/K) w        (E^T E = I, E^T J E = 11^T)
 *
 * so `R = lambda (I - 11^T/K)`. Penalising the raw `[w; 0]` embedding instead gives
 * `R = lambda I`. Both are symmetric positive definite -- the centered form has eigenvalues
 * `lambda` and `lambda/K` -- so `H + R` stays positive definite either way.
 *
 * The centered form is the default because it makes the fitted model invariant to which
 * class is chosen as the reference, which `lambda I` is not. It is also, up to the factor
 * `2 lambda`, the same projection that appears in Boehning's uniform Loewner bound
 * `diag(p) - p p^T <= (1/2)(I - 11^T/K)` for multinomial logistic: both come from removing
 * the all-ones gauge direction.
 *
 * `n_classes == 0` selects the raw gauge, `R = lambda I`.
 */
struct ExactL2 {
  double lambda{0.0};
  bst_target_t n_classes{0};

  [[nodiscard]] static ExactL2 Centered(double lambda, bst_target_t n_classes) {
    return ExactL2{lambda, n_classes};
  }
  [[nodiscard]] static ExactL2 Raw(double lambda) { return ExactL2{lambda, 0}; }

  [[nodiscard]] bool IsCentered() const { return n_classes != 0; }
  [[nodiscard]] double Diagonal() const {
    return IsCentered() ? lambda * (1.0 - 1.0 / static_cast<double>(n_classes)) : lambda;
  }
  [[nodiscard]] double OffDiagonal() const {
    return IsCentered() ? -lambda / static_cast<double>(n_classes) : 0.0;
  }
};

/**
 * @brief Compute `y = (H + lambda I) x` for a packed symmetric Hessian.
 *
 * Used to measure the residual of a solved Newton system.
 */
inline void PackedSymv(PackedMultinomialStats<double const> stats, double lambda,
                       Span<double const> x, Span<double> y) {
  auto n = stats.NumFree();
  CHECK_EQ(x.size(), n);
  CHECK_EQ(y.size(), n);
  for (std::size_t i = 0; i < n; ++i) {
    double acc = lambda * x[i];
    for (std::size_t j = 0; j < n; ++j) {
      acc += stats.GetHessian(i, j) * x[j];
    }
    y[i] = acc;
  }
}

/**
 * @brief Exact Newton step for one leaf of a multinomial logistic model.
 *
 * Solves the regularized system
 *
 *   (H + lambda I) w = -G
 *
 * where `G` and `H` are the accumulated gradient and dense Hessian over the leaf, written
 * in the `K - 1` free logits of @ref PackedMultinomialStats.
 *
 * `H` is symmetric positive semi-definite, being a sum of per-sample `diag(p) - p p^T`
 * blocks, so `H + lambda I` is positive definite for `lambda > 0`. The system is solved
 * with a square root free Cholesky factorization, `A = L D L^T` with unit lower triangular
 * `L`, followed by forward and backward substitution. The inverse is never formed: it
 * would cost more, lose accuracy, and discard the triangular structure.
 *
 * The factorization works on the same packed lower triangle as the statistics, so only the
 * lower triangle is ever read and symmetry needs no separate handling. Scratch space is
 * allocated once per solver and reused for every leaf.
 */
class ExactMultinomialLeafSolver {
 public:
  /**
   * @param n_free Number of free logits, `K - 1` for `K` classes.
   */
  explicit ExactMultinomialLeafSolver(std::size_t n_free)
      : n_free_{n_free}, factor_(PackedHessianSize(n_free), 0.0) {}

  [[nodiscard]] std::size_t NumFree() const { return n_free_; }

  /**
   * @brief Solve `(H + R) w = -G` and, optionally, report the gain.
   *
   * @param stats    Accumulated leaf gradient and Hessian.
   * @param reg      L2 regularization, see @ref ExactL2.
   * @param out_w    Output leaf weight, `n_free` entries.
   * @param out_gain If non-null, receives `G^T (H + R)^-1 G`, which is XGBoost's gain
   *                 convention (twice the loss reduction, see @ref ExactLeafGain). It is
   *                 obtained from the same factorization at no extra cost.
   *
   * @return False when `H + R` is not numerically positive definite, in which case
   *         `out_w` and `out_gain` are left untouched.
   */
  [[nodiscard]] bool Solve(PackedMultinomialStats<double const> stats, ExactL2 reg,
                           Span<double> out_w, double* out_gain = nullptr) {
    CHECK_EQ(stats.NumFree(), n_free_);
    CHECK_EQ(out_w.size(), n_free_);
    CHECK_GE(reg.lambda, 0.0);

    if (!this->Factorize(stats, reg)) {
      return false;
    }

    for (std::size_t i = 0; i < n_free_; ++i) {
      out_w[i] = -stats.GetGradient(i);
    }
    // Forward substitution, L z = b. L has an implicit unit diagonal.
    for (std::size_t i = 0; i < n_free_; ++i) {
      double acc = out_w[i];
      for (std::size_t k = 0; k < i; ++k) {
        acc -= factor_[PackedHessianIndex(i, k)] * out_w[k];
      }
      out_w[i] = acc;
    }
    // G^T A^-1 G = (L^-1 G)^T D^-1 (L^-1 G), and out_w currently holds -L^-1 G, whose sign
    // cancels under the square. Reading it here avoids a second triangular solve, and the
    // sum is non-negative because every pivot is positive.
    if (out_gain) {
      double gain = 0.0;
      for (std::size_t i = 0; i < n_free_; ++i) {
        gain += out_w[i] * out_w[i] / factor_[PackedHessianIndex(i, i)];
      }
      *out_gain = gain;
    }
    // Diagonal solve, D y = z.
    for (std::size_t i = 0; i < n_free_; ++i) {
      out_w[i] /= factor_[PackedHessianIndex(i, i)];
    }
    // Backward substitution, L^T w = y.
    for (std::size_t i = n_free_; i-- > 0;) {
      double acc = out_w[i];
      for (std::size_t k = i + 1; k < n_free_; ++k) {
        acc -= factor_[PackedHessianIndex(k, i)] * out_w[k];
      }
      out_w[i] = acc;
    }
    return true;
  }

  /**
   * @brief `G^T (H + R)^-1 G` without computing the weight.
   *
   * Split enumeration needs only the gain, and the gain falls out of the forward
   * substitution: with `(H+R) = L D L^T` and `z = L^-1 G`, it is `sum_i z_i^2 / d_i`. The
   * diagonal solve and the backward substitution exist solely to turn `z` into `w`, so for a
   * gain-only caller they are pure overhead.
   *
   * @return False when `H + R` is not positive definite, leaving @p out_gain untouched.
   */
  [[nodiscard]] bool Gain(PackedMultinomialStats<double const> stats, ExactL2 reg,
                          double* out_gain) {
    CHECK_EQ(stats.NumFree(), n_free_);
    CHECK_GE(reg.lambda, 0.0);
    if (!this->Factorize(stats, reg)) {
      return false;
    }
    scratch_.resize(n_free_);
    double gain = 0.0;
    for (std::size_t i = 0; i < n_free_; ++i) {
      double acc = stats.GetGradient(i);
      for (std::size_t k = 0; k < i; ++k) {
        acc -= factor_[PackedHessianIndex(i, k)] * scratch_[k];
      }
      scratch_[i] = acc;
      gain += acc * acc / factor_[PackedHessianIndex(i, i)];
    }
    *out_gain = gain;
    return true;
  }

  /** @brief Convenience overload using the raw gauge, `R = lambda I`. */
  [[nodiscard]] bool Solve(PackedMultinomialStats<double const> stats, double lambda,
                           Span<double> out_w) {
    return this->Solve(stats, ExactL2::Raw(lambda), out_w);
  }

 private:
  /**
   * @brief Factor `H + R` as `L D L^T` in place over the packed triangle.
   *
   * After the call the diagonal holds `D` and the strict lower triangle holds `L`.
   */
  [[nodiscard]] bool Factorize(PackedMultinomialStats<double const> stats, ExactL2 reg) {
    auto on_diagonal = reg.Diagonal();
    auto off_diagonal = reg.OffDiagonal();
    for (std::size_t i = 0; i < n_free_; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        auto v = stats.GetHessian(i, j);
        factor_[PackedHessianIndex(i, j)] = i == j ? v + on_diagonal : v + off_diagonal;
      }
    }

    for (std::size_t j = 0; j < n_free_; ++j) {
      double d = factor_[PackedHessianIndex(j, j)];
      for (std::size_t k = 0; k < j; ++k) {
        auto l_jk = factor_[PackedHessianIndex(j, k)];
        d -= l_jk * l_jk * factor_[PackedHessianIndex(k, k)];
      }
      // A non-positive pivot means the matrix is singular or indefinite, which the exact
      // multinomial Hessian can reach when a class probability underflows to zero.
      if (!(d > 0.0) || !std::isfinite(d)) {
        return false;
      }
      factor_[PackedHessianIndex(j, j)] = d;

      for (std::size_t i = j + 1; i < n_free_; ++i) {
        double acc = factor_[PackedHessianIndex(i, j)];
        for (std::size_t k = 0; k < j; ++k) {
          acc -= factor_[PackedHessianIndex(i, k)] * factor_[PackedHessianIndex(k, k)] *
                 factor_[PackedHessianIndex(j, k)];
        }
        factor_[PackedHessianIndex(i, j)] = acc / d;
      }
    }
    return true;
  }

  std::size_t n_free_;
  std::vector<double> factor_;
  std::vector<double> scratch_;
};
}  // namespace xgboost::common
#endif  // XGBOOST_COMMON_EXACT_MULTINOMIAL_LEAF_SOLVER_H_
