/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <algorithm>  // for max
#include <cmath>      // for fabs
#include <cstddef>    // for size_t
#include <vector>     // for vector

#include "../../src/common/exact_multinomial/leaf_solver.h"
#include "../../src/common/exact_multinomial/packed_stats.h"

namespace xgboost::common {
namespace {
// Owns the packed buffer a leaf statistic views.
class LeafStats {
 public:
  explicit LeafStats(std::size_t n_free)
      : n_free_{n_free}, buffer_(PackedStatsStride(n_free), 0.0) {}

  [[nodiscard]] PackedMultinomialStats<double> View() {
    return {Span<double>{buffer_.data(), buffer_.size()}, n_free_};
  }
  /** @brief Fill the Hessian with `diag(p) - p p^T` over the free classes. */
  void SetSoftmaxHessian(std::vector<double> const& p, double weight) {
    auto stats = this->View();
    for (std::size_t i = 0; i < n_free_; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        stats.SetHessian(i, j, weight * ((i == j ? p[i] : 0.0) - p[i] * p[j]));
      }
    }
  }

 private:
  std::size_t n_free_;
  std::vector<double> buffer_;
};

// Largest absolute entry of `(H + lambda I) w + g`, computed independently of the solver.
[[nodiscard]] double MaxResidual(PackedMultinomialStats<double const> stats, double lambda,
                                 std::vector<double> const& w) {
  std::vector<double> lhs(w.size(), 0.0);
  PackedSymv(stats, lambda, Span<double const>{w.data(), w.size()},
             Span<double>{lhs.data(), lhs.size()});
  double worst = 0.0;
  for (std::size_t i = 0; i < w.size(); ++i) {
    worst = std::max(worst, std::fabs(lhs[i] + stats.GetGradient(i)));
  }
  return worst;
}

[[nodiscard]] double MaxAbs(std::vector<double> const& v) {
  double worst = 0.0;
  for (auto e : v) {
    worst = std::max(worst, std::fabs(e));
  }
  return worst;
}
}  // anonymous namespace

TEST(ExactMultinomialSolver, Known2x2System) {
  // Three classes with class 2 as the reference, p = [0.1, 0.3, 0.6]:
  //   H = [[0.09, -0.03], [-0.03, 0.21]]
  LeafStats leaf{2};
  leaf.SetSoftmaxHessian({0.1, 0.3, 0.6}, 1.0);
  auto stats = leaf.View();
  stats.SetGradient(0, 0.1);
  stats.SetGradient(1, -0.2);

  double constexpr kLambda = 0.01;
  std::vector<double> w(2, 0.0);
  ExactMultinomialLeafSolver solver{2};
  ASSERT_TRUE(solver.Solve(stats, kLambda, Span<double>{w.data(), w.size()}));

  // Closed form 2x2 inverse, an independent path to the same answer.
  auto a00 = stats.GetHessian(0, 0) + kLambda;
  auto a01 = stats.GetHessian(0, 1);
  auto a11 = stats.GetHessian(1, 1) + kLambda;
  auto det = a00 * a11 - a01 * a01;
  ASSERT_DOUBLE_EQ(det, 0.0211);
  auto expected_0 = (-a11 * stats.GetGradient(0) + a01 * stats.GetGradient(1)) / det;
  auto expected_1 = (a01 * stats.GetGradient(0) - a00 * stats.GetGradient(1)) / det;

  ASSERT_NEAR(w[0], expected_0, 1e-14);
  ASSERT_NEAR(w[1], expected_1, 1e-14);
  // -0.016 / 0.0211 and 0.017 / 0.0211, worked out by hand.
  ASSERT_NEAR(w[0], -0.7582938388625592, 1e-14);
  ASSERT_NEAR(w[1], 0.8056872037914692, 1e-14);
}

TEST(ExactMultinomialSolver, ResidualIsZero) {
  // Five classes, class 4 is the reference, scaled by the number of samples in the leaf.
  std::vector<double> p{0.05, 0.1, 0.15, 0.3, 0.4};
  std::vector<double> counts{3.0, 12.0, 18.0, 27.0};
  double constexpr kNumSamples = 100.0;
  auto n_free = NumFreeClasses(p.size());

  LeafStats leaf{n_free};
  leaf.SetSoftmaxHessian(p, kNumSamples);
  auto stats = leaf.View();
  for (std::size_t i = 0; i < n_free; ++i) {
    stats.SetGradient(i, kNumSamples * p[i] - counts[i]);
  }

  double constexpr kLambda = 1.0;
  std::vector<double> w(n_free, 0.0);
  ExactMultinomialLeafSolver solver{n_free};
  ASSERT_TRUE(solver.Solve(stats, kLambda, Span<double>{w.data(), w.size()}));

  auto scale = std::max(1.0, kNumSamples * MaxAbs(w));
  ASSERT_LT(MaxResidual(stats.AsConst(), kLambda, w), 1e-12 * scale);

  // A solver instance is reusable across leaves, so a second solve must not drift.
  std::vector<double> again(n_free, 0.0);
  ASSERT_TRUE(solver.Solve(stats, kLambda, Span<double>{again.data(), again.size()}));
  for (std::size_t i = 0; i < n_free; ++i) {
    ASSERT_EQ(w[i], again[i]);
  }
}

TEST(ExactMultinomialSolver, UpperAndLowerTriangleIndicesAgree) {
  std::size_t constexpr kNumFree = 3;
  std::vector<double> gradient{0.4, -0.25, 0.7};
  // The same matrix written twice: once through lower triangle indices, once through upper.
  double const dense[kNumFree][kNumFree] = {
      {1.5, -0.4, 0.2}, {-0.4, 2.25, -0.75}, {0.2, -0.75, 3.0}};

  LeafStats lower_leaf{kNumFree};
  auto lower = lower_leaf.View();
  LeafStats upper_leaf{kNumFree};
  auto upper = upper_leaf.View();
  for (std::size_t i = 0; i < kNumFree; ++i) {
    lower.SetGradient(i, gradient[i]);
    upper.SetGradient(i, gradient[i]);
    for (std::size_t j = 0; j <= i; ++j) {
      lower.SetHessian(i, j, dense[i][j]);
      upper.SetHessian(j, i, dense[j][i]);
    }
  }
  // Mirrored writes land on the same packed scalars.
  for (std::size_t k = 0; k < PackedHessianSize(kNumFree); ++k) {
    ASSERT_EQ(lower.Hessian()[k], upper.Hessian()[k]);
  }

  double constexpr kLambda = 0.5;
  ExactMultinomialLeafSolver solver{kNumFree};
  std::vector<double> w_lower(kNumFree, 0.0);
  std::vector<double> w_upper(kNumFree, 0.0);
  ASSERT_TRUE(solver.Solve(lower, kLambda, Span<double>{w_lower.data(), w_lower.size()}));
  ASSERT_TRUE(solver.Solve(upper, kLambda, Span<double>{w_upper.data(), w_upper.size()}));
  for (std::size_t i = 0; i < kNumFree; ++i) {
    ASSERT_EQ(w_lower[i], w_upper[i]);
  }

  // The solve uses the full symmetric matrix, so the dense residual is zero as well.
  for (std::size_t i = 0; i < kNumFree; ++i) {
    double acc = kLambda * w_lower[i];
    for (std::size_t j = 0; j < kNumFree; ++j) {
      acc += dense[i][j] * w_lower[j];
    }
    ASSERT_NEAR(acc, -gradient[i], 1e-14);
  }
}

TEST(ExactMultinomialSolver, DiagonalSystemMatchesClosedForm) {
  // Diagonal system, so each weight is -g_i / (h_ii + lambda) by hand.
  {
    LeafStats leaf{3};
    auto stats = leaf.View();
    std::vector<double> diag{2.0, 4.0, 8.0};
    std::vector<double> gradient{3.0, -6.0, 12.0};
    for (std::size_t i = 0; i < 3; ++i) {
      stats.SetGradient(i, gradient[i]);
      stats.SetHessian(i, i, diag[i]);
    }

    std::vector<double> w(3, 0.0);
    ExactMultinomialLeafSolver solver{3};
    ASSERT_TRUE(solver.Solve(stats, 1.0, Span<double>{w.data(), w.size()}));
    ASSERT_NEAR(w[0], -1.0, 1e-15);         // -3 / 3
    ASSERT_NEAR(w[1], 1.2, 1e-15);          //  6 / 5
    ASSERT_NEAR(w[2], -12.0 / 9.0, 1e-15);  // -12 / 9
  }
  // Coupled system with an exact rational solution: A = [[4, 1], [1, 3]], -g = [1, 2],
  // det = 11, so w = [1/11, 7/11].
  {
    LeafStats leaf{2};
    auto stats = leaf.View();
    stats.SetGradient(0, -1.0);
    stats.SetGradient(1, -2.0);
    stats.SetHessian(0, 0, 4.0);
    stats.SetHessian(1, 0, 1.0);
    stats.SetHessian(1, 1, 3.0);

    std::vector<double> w(2, 0.0);
    ExactMultinomialLeafSolver solver{2};
    ASSERT_TRUE(solver.Solve(stats, 0.0, Span<double>{w.data(), w.size()}));
    ASSERT_NEAR(w[0], 1.0 / 11.0, 1e-15);
    ASSERT_NEAR(w[1], 7.0 / 11.0, 1e-15);
  }
}

TEST(ExactMultinomialSolver, NearSingularHessianStaysAccurate) {
  // A class that carries almost no mass leaves the Hessian close to singular.
  std::vector<double> p{1e-6, 0.2, 0.3, 0.4, 0.099999};
  double total = 0.0;
  for (auto e : p) {
    total += e;
  }
  for (auto& e : p) {
    e /= total;
  }
  auto n_free = NumFreeClasses(p.size());

  LeafStats leaf{n_free};
  leaf.SetSoftmaxHessian(p, 1.0);
  auto stats = leaf.View();
  std::vector<double> gradient{0.3, -0.1, 0.05, -0.25};
  for (std::size_t i = 0; i < n_free; ++i) {
    stats.SetGradient(i, gradient[i]);
  }

  double constexpr kTinyLambda = 1e-8;
  std::vector<double> w(n_free, 0.0);
  ExactMultinomialLeafSolver solver{n_free};
  ASSERT_TRUE(solver.Solve(stats, kTinyLambda, Span<double>{w.data(), w.size()}));
  // Backward stable: the residual stays at rounding level relative to ||A|| * ||w||.
  auto scale = std::max(1.0, MaxAbs(w));
  ASSERT_LT(MaxResidual(stats.AsConst(), kTinyLambda, w), 1e-12 * scale);

  // An exactly singular Hessian: class 0 carries no mass at all.
  LeafStats singular_leaf{2};
  singular_leaf.SetSoftmaxHessian({0.0, 0.4, 0.6}, 1.0);
  auto singular = singular_leaf.View();
  singular.SetGradient(0, 0.5);
  singular.SetGradient(1, -0.5);
  ASSERT_DOUBLE_EQ(singular.GetHessian(0, 0), 0.0);

  std::vector<double> untouched{42.0, 42.0};
  ExactMultinomialLeafSolver small_solver{2};
  ASSERT_FALSE(small_solver.Solve(singular, 0.0, Span<double>{untouched.data(), untouched.size()}));
  ASSERT_EQ(untouched[0], 42.0);
  ASSERT_EQ(untouched[1], 42.0);

  // Regularization restores positive definiteness.
  ASSERT_TRUE(small_solver.Solve(singular, 1e-3, Span<double>{untouched.data(), untouched.size()}));
  ASSERT_LT(MaxResidual(singular.AsConst(), 1e-3, untouched),
            1e-12 * std::max(1.0, MaxAbs(untouched)));

  // An indefinite matrix is rejected rather than silently producing a maximizer.
  LeafStats indefinite_leaf{2};
  auto indefinite = indefinite_leaf.View();
  indefinite.SetHessian(0, 0, 1.0);
  indefinite.SetHessian(1, 1, -1.0);
  indefinite.SetGradient(0, 1.0);
  indefinite.SetGradient(1, 1.0);
  std::vector<double> rejected(2, 0.0);
  ASSERT_FALSE(small_solver.Solve(indefinite, 0.0, Span<double>{rejected.data(), rejected.size()}));
}
}  // namespace xgboost::common
