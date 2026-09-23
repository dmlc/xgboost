/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>

#include <algorithm>  // for max, max_element
#include <cmath>      // for exp, fabs, isfinite, cbrt, sqrt, log
#include <cstddef>    // for size_t
#include <cstdio>     // for printf
#include <limits>     // for numeric_limits
#include <numeric>    // for accumulate
#include <random>     // for mt19937, uniform_real_distribution
#include <vector>     // for vector

#include "../../src/common/exact_multinomial/leaf_solver.h"
#include "../../src/common/exact_multinomial/packed_stats.h"

namespace xgboost::common {
namespace {
/** @brief Numerically stable softmax of `logits`. */
std::vector<double> Softmax(std::vector<double> logits) {
  auto m = *std::max_element(logits.cbegin(), logits.cend());
  double total = 0.0;
  for (auto& v : logits) {
    v = std::exp(v - m);
    total += v;
  }
  for (auto& v : logits) {
    v /= total;
  }
  return logits;
}

/**
 * @brief Owning packed statistics, accumulated from probabilities and weights.
 *
 * Written independently of the production accumulation path so that the two can disagree.
 */
class Stats {
 public:
  explicit Stats(std::size_t n_free) : n_free_{n_free}, buffer_(PackedStatsStride(n_free), 0.0) {}
  Stats(std::vector<double> const& p, std::size_t label, double w) : Stats{p.size() - 1} {
    this->Add(p, label, w);
  }

  void Add(std::vector<double> const& p, std::size_t label, double w) {
    auto stats = this->View();
    for (std::size_t i = 0; i < n_free_; ++i) {
      stats.AddGradient(i, w * (p[i] - (label == i ? 1.0 : 0.0)));
      for (std::size_t j = 0; j <= i; ++j) {
        stats.AddHessian(i, j, w * p[i] * ((i == j ? 1.0 : 0.0) - p[j]));
      }
    }
  }

  [[nodiscard]] PackedMultinomialStats<double> View() {
    return {Span<double>{buffer_.data(), buffer_.size()}, n_free_};
  }
  [[nodiscard]] PackedMultinomialStats<double const> ConstView() const {
    return {Span<double const>{buffer_.data(), buffer_.size()}, n_free_};
  }

 private:
  std::size_t n_free_;
  std::vector<double> buffer_;
};

/** @brief `dL(w) = G^T w + 0.5 w^T (H + R) w`, evaluated without the solver. */
double DeltaL(PackedMultinomialStats<double const> stats, ExactL2 reg,
              std::vector<double> const& w) {
  auto d = stats.NumFree();
  double linear = 0.0;
  for (std::size_t i = 0; i < d; ++i) {
    linear += stats.GetGradient(i) * w[i];
  }
  double quad = 0.0;
  for (std::size_t i = 0; i < d; ++i) {
    for (std::size_t j = 0; j < d; ++j) {
      auto a = stats.GetHessian(i, j) + (i == j ? reg.Diagonal() : reg.OffDiagonal());
      quad += w[i] * a * w[j];
    }
  }
  return linear + 0.5 * quad;
}

/** @brief Dense row-major `H + R`, for linear algebra that does not reuse the solver. */
std::vector<double> DenseSystem(PackedMultinomialStats<double const> stats, ExactL2 reg) {
  auto d = stats.NumFree();
  std::vector<double> a(d * d, 0.0);
  for (std::size_t i = 0; i < d; ++i) {
    for (std::size_t j = 0; j < d; ++j) {
      a[i * d + j] = stats.GetHessian(i, j) + (i == j ? reg.Diagonal() : reg.OffDiagonal());
    }
  }
  return a;
}

/** @brief `max_i |G_i + (A w)_i|`, the residual of the solved Newton system. */
double NewtonResidual(PackedMultinomialStats<double const> stats, std::vector<double> const& a,
                      std::vector<double> const& w) {
  auto d = stats.NumFree();
  double residual = 0.0;
  for (std::size_t i = 0; i < d; ++i) {
    double acc = stats.GetGradient(i);
    for (std::size_t j = 0; j < d; ++j) {
      acc += a[i * d + j] * w[j];
    }
    residual = std::max(residual, std::fabs(acc));
  }
  return residual;
}

double MaxAbs(std::vector<double> const& v) {
  double m = 0.0;
  for (auto x : v) {
    m = std::max(m, std::fabs(x));
  }
  return m;
}

/**
 * @brief Smallest eigenvalue of a symmetric matrix, by power iteration on `cI - A`.
 *
 * `c` is a Gershgorin bound on the largest eigenvalue, so `cI - A` is positive semi-definite
 * and its dominant eigenvalue is `c - lambda_min`.
 */
double SmallestEigenvalue(std::vector<double> const& a, std::size_t d) {
  double shift = 0.0;
  for (std::size_t i = 0; i < d; ++i) {
    double row = 0.0;
    for (std::size_t j = 0; j < d; ++j) {
      row += std::fabs(a[i * d + j]);
    }
    shift = std::max(shift, row);
  }
  std::vector<double> v(d, 1.0 / std::sqrt(static_cast<double>(d)));
  std::vector<double> next(d, 0.0);
  double dominant = 0.0;
  for (int iter = 0; iter < 5000; ++iter) {
    for (std::size_t i = 0; i < d; ++i) {
      double acc = shift * v[i];
      for (std::size_t j = 0; j < d; ++j) {
        acc -= a[i * d + j] * v[j];
      }
      next[i] = acc;
    }
    double norm = 0.0;
    for (auto x : next) {
      norm += x * x;
    }
    norm = std::sqrt(norm);
    if (norm == 0.0) {
      return shift;
    }
    for (std::size_t i = 0; i < d; ++i) {
      v[i] = next[i] / norm;
    }
    dominant = norm;
  }
  return shift - dominant;
}

std::vector<double> UniformP(std::size_t k) { return std::vector<double>(k, 1.0 / k); }

ExactL2 Centered(double lambda, std::size_t k) {
  return ExactL2::Centered(lambda, static_cast<bst_target_t>(k));
}
}  // anonymous namespace

// --------------------------------------------------------------------------- //
// Phase 1 - Gain() must agree with Solve(), which is the whole point of having both.
// --------------------------------------------------------------------------- //

TEST(ExactNumerics, GainMatchesSolveDerivedGain) {
  std::mt19937 rng{20260922};
  std::uniform_real_distribution<double> logit{-6.0, 6.0};

  for (std::size_t k : {2ul, 3ul, 7ul, 10ul}) {
    auto d = k - 1;
    ExactMultinomialLeafSolver solver{d};
    for (double lambda : {0.0, 1e-8, 1e-3, 1.0, 100.0}) {
      auto reg = Centered(lambda, k);
      for (int trial = 0; trial < 40; ++trial) {
        // Several rows, so H is a genuine sum of outer products rather than rank one.
        Stats stats{d};
        for (int row = 0; row < 5; ++row) {
          std::vector<double> z(k);
          for (auto& v : z) {
            v = logit(rng);
          }
          stats.Add(Softmax(z), static_cast<std::size_t>(trial) % k, 0.5 + 0.5 * row);
        }
        auto view = stats.ConstView();

        double gain_only = 0.0;
        auto gain_ok = solver.Gain(view, reg, &gain_only);

        std::vector<double> w(d, 0.0);
        double gain_from_solve = 0.0;
        auto solve_ok = solver.Solve(view, reg, Span<double>{w.data(), w.size()}, &gain_from_solve);

        // The two entry points share a factorization, so they must also share its verdict.
        ASSERT_EQ(gain_ok, solve_ok) << "K=" << k << " lambda=" << lambda;
        if (!gain_ok) {
          continue;
        }
        auto scale = std::max(1.0, std::fabs(gain_from_solve));
        EXPECT_NEAR(gain_only, gain_from_solve, 1e-10 * scale)
            << "K=" << k << " lambda=" << lambda;
        // Both must equal -2 * dL(w*), computed from the dense system by a third path.
        EXPECT_NEAR(gain_only, -2.0 * DeltaL(view, reg, w), 1e-8 * scale)
            << "K=" << k << " lambda=" << lambda;
        EXPECT_GE(gain_only, -1e-12) << "gain must be non-negative";
      }
    }
  }
}

// --------------------------------------------------------------------------- //
// Phase 2 / 11 / 12 - conditioning, pivot failure, regularization limits
// --------------------------------------------------------------------------- //

TEST(ExactNumerics, ExtremeConditioning) {
  std::size_t constexpr kNumClasses = 5;
  auto d = kNumClasses - 1;
  ExactMultinomialLeafSolver solver{d};

  std::vector<std::vector<double>> distributions{
      UniformP(kNumClasses),
      {0.9999, 1e-4, 1e-5, 1e-6, 1e-7},
      {1e-12, 0.25, 0.25, 0.25, 0.25 - 1e-12},
      {1.0 - 4e-16, 1e-16, 1e-16, 1e-16, 1e-16},
  };

  for (auto p : distributions) {
    auto total = std::accumulate(p.cbegin(), p.cend(), 0.0);
    for (auto& v : p) {
      v /= total;
    }
    for (double lambda : {0.0, 1e-12, 1e-6, 1.0, 1e6}) {
      Stats stats{p, 0, 1.0};
      auto view = stats.ConstView();
      auto reg = Centered(lambda, kNumClasses);

      std::vector<double> w(d, 0.0);
      double gain = 0.0;
      if (!solver.Solve(view, reg, Span<double>{w.data(), w.size()}, &gain)) {
        // A refusal is a legitimate outcome here; what matters is that nothing leaks out
        // of it, which InvalidCurvatureIsRejectedNotApproximated checks directly.
        continue;
      }
      for (auto v : w) {
        ASSERT_TRUE(std::isfinite(v)) << "non-finite weight, lambda=" << lambda;
      }
      ASSERT_TRUE(std::isfinite(gain)) << "non-finite gain, lambda=" << lambda;
      EXPECT_GE(gain, -1e-9);

      // Backward stability: the residual scales with ||A|| ||w||, not with cond(A).
      auto a = DenseSystem(view, reg);
      EXPECT_LT(NewtonResidual(view, a, w), 1e-9 * std::max(1.0, MaxAbs(a) * MaxAbs(w)))
          << "lambda=" << lambda;
    }
  }
}

TEST(ExactNumerics, InvalidCurvatureIsRejectedNotApproximated) {
  std::size_t constexpr kNumClasses = 3;
  auto d = kNumClasses - 1;
  ExactMultinomialLeafSolver solver{d};
  std::vector<double> w(d, 42.0);
  double gain = -1.0;

  // A class with exactly zero probability gives a zero row and column.
  Stats singular({0.0, 0.4, 0.6}, 1, 1.0);
  EXPECT_FALSE(solver.Solve(singular.ConstView(), ExactL2::Raw(0.0),
                            Span<double>{w.data(), w.size()}, &gain));
  EXPECT_FALSE(solver.Gain(singular.ConstView(), ExactL2::Raw(0.0), &gain));
  // Nothing partial and nothing non-finite escapes a refused factorization.
  for (auto v : w) {
    EXPECT_EQ(v, 42.0);
  }
  EXPECT_EQ(gain, -1.0);

  // An indefinite matrix is rejected rather than returning a saddle point as a maximum.
  Stats indefinite{d};
  indefinite.View().SetHessian(0, 0, 1.0);
  indefinite.View().SetHessian(1, 0, 0.0);
  indefinite.View().SetHessian(1, 1, -1.0);
  EXPECT_FALSE(solver.Gain(indefinite.ConstView(), ExactL2::Raw(0.0), &gain));
  EXPECT_EQ(gain, -1.0);

  // A zero-information node: no gradient, valid curvature. This is a *valid* candidate with
  // zero gain, and it must stay distinguishable from the refusals above.
  Stats flat{UniformP(kNumClasses), 0, 1.0};
  auto view = flat.View();
  for (std::size_t i = 0; i < d; ++i) {
    view.SetGradient(i, 0.0);
  }
  double zero_gain = -1.0;
  EXPECT_TRUE(solver.Gain(flat.ConstView(), Centered(1.0, kNumClasses), &zero_gain));
  EXPECT_NEAR(zero_gain, 0.0, 1e-15);
}

TEST(ExactNumerics, RegularizationShrinksGainMonotonically) {
  std::size_t constexpr kNumClasses = 4;
  auto d = kNumClasses - 1;
  ExactMultinomialLeafSolver solver{d};
  Stats stats{d};
  stats.Add({0.1, 0.2, 0.3, 0.4}, 0, 1.0);
  stats.Add({0.4, 0.3, 0.2, 0.1}, 2, 2.0);
  auto view = stats.ConstView();

  double unregularized_norm = 0.0;
  double previous_gain = std::numeric_limits<double>::infinity();
  double last_norm = 0.0;
  for (double lambda : {0.0, 1e-3, 1.0, 100.0, 1e6}) {
    auto reg = Centered(lambda, kNumClasses);
    std::vector<double> w(d, 0.0);
    double gain = 0.0;
    ASSERT_TRUE(solver.Solve(view, reg, Span<double>{w.data(), w.size()}, &gain))
        << "lambda=" << lambda;
    // R(lambda) grows in the Loewner order, so (H+R)^-1 shrinks and so does the gain. The
    // same argument does NOT extend to ||w||: inversion is operator monotone but squaring
    // is not, so no per-step claim is made about the step norm.
    EXPECT_LE(gain, previous_gain + 1e-12) << "lambda=" << lambda;
    previous_gain = gain;
    last_norm = MaxAbs(w);
    if (lambda == 0.0) {
      unregularized_norm = last_norm;
    }
    // Both eigenvalues of the centered R are positive (lambda and lambda/K), so H + R is
    // positive definite even where H alone is only semi-definite.
    if (lambda > 0.0) {
      EXPECT_GT(SmallestEigenvalue(DenseSystem(view, reg), d), 0.0) << "lambda=" << lambda;
    }
  }
  // lambda = 1e6 against an O(1) gradient must drive the step to essentially zero.
  EXPECT_LT(last_norm, 1e-4 * unregularized_norm);
}

// --------------------------------------------------------------------------- //
// Phase 3 - the closed forms, against finite differences of the loss itself
// --------------------------------------------------------------------------- //

TEST(ExactNumerics, GradientAndHessianAgainstFiniteDifferences) {
  // Central differences balance O(h^2) truncation against O(eps/h) round-off near
  // h = eps^(1/3). That is derived, not tuned to make a particular error figure appear.
  auto const step = std::cbrt(std::numeric_limits<double>::epsilon());

  for (std::size_t k : {2ul, 3ul, 5ul, 7ul}) {
    auto d = k - 1;
    std::size_t label = k / 2;

    // -log p[label], in the K-1 free logits with the last class pinned to zero.
    auto objective = [&](std::vector<double> const& free_logits) {
      std::vector<double> full(free_logits);
      full.push_back(0.0);
      auto m = *std::max_element(full.cbegin(), full.cend());
      double lse = 0.0;
      for (auto v : full) {
        lse += std::exp(v - m);
      }
      return m + std::log(lse) - full[label];
    };
    auto probabilities = [&](std::vector<double> const& free_logits) {
      std::vector<double> full(free_logits);
      full.push_back(0.0);
      return Softmax(full);
    };

    std::vector<double> z(d);
    for (std::size_t i = 0; i < d; ++i) {
      z[i] = 0.4 * static_cast<double>(i) - 0.7;
    }
    auto p = probabilities(z);

    double worst_grad = 0.0;
    double worst_hess = 0.0;
    for (std::size_t i = 0; i < d; ++i) {
      auto zp = z;
      auto zm = z;
      zp[i] += step;
      zm[i] -= step;
      auto numerical = (objective(zp) - objective(zm)) / (2.0 * step);
      auto analytic = p[i] - (label == i ? 1.0 : 0.0);
      worst_grad = std::max(worst_grad, std::fabs(numerical - analytic));

      auto pp = probabilities(zp);
      auto pm = probabilities(zm);
      for (std::size_t j = 0; j < d; ++j) {
        // dg_j/dz_i, where g_j = p_j - y_j.
        auto numerical_h = (pp[j] - pm[j]) / (2.0 * step);
        auto analytic_h = p[j] * ((i == j ? 1.0 : 0.0) - p[i]);
        worst_hess = std::max(worst_hess, std::fabs(numerical_h - analytic_h));
      }
    }
    std::printf("  K=%zu finite differences: grad %.3e, hess %.3e (step %.3e)\n", k, worst_grad,
                worst_hess, step);
    EXPECT_LT(worst_grad, 1e-8) << "K=" << k;
    EXPECT_LT(worst_hess, 1e-7) << "K=" << k;
  }
}

// --------------------------------------------------------------------------- //
// Phase 4 / 5 - packed round-trip and structural invariants
// --------------------------------------------------------------------------- //

TEST(ExactNumerics, PackedRoundTripAndStructure) {
  std::mt19937 rng{7};
  std::uniform_real_distribution<double> logit{-4.0, 4.0};

  for (std::size_t k : {2ul, 3ul, 7ul, 10ul, 20ul}) {
    auto d = k - 1;
    for (int trial = 0; trial < 8; ++trial) {
      std::vector<double> z(k);
      for (auto& v : z) {
        v = logit(rng);
      }
      auto p = Softmax(z);
      double weight = 0.25 + static_cast<double>(trial);
      auto label = static_cast<std::size_t>(trial) % k;

      Stats stats{p, label, weight};
      auto view = stats.ConstView();

      double worst = 0.0;
      for (std::size_t i = 0; i < d; ++i) {
        for (std::size_t j = 0; j < d; ++j) {
          auto expected = weight * p[i] * ((i == j ? 1.0 : 0.0) - p[j]);
          worst = std::max(worst, std::fabs(view.GetHessian(i, j) - expected));
          // Symmetry is exact, not approximate: both orderings address one scalar.
          ASSERT_EQ(view.GetHessian(i, j), view.GetHessian(j, i)) << "K=" << k;
        }
      }
      EXPECT_LT(worst, 1e-15 * std::max(1.0, weight)) << "K=" << k;

      // Diagonal strictly positive, off-diagonal strictly negative.
      for (std::size_t i = 0; i < d; ++i) {
        EXPECT_GT(view.GetHessian(i, i), 0.0) << "K=" << k;
        for (std::size_t j = 0; j < i; ++j) {
          EXPECT_LT(view.GetHessian(i, j), 0.0) << "K=" << k;
        }
      }

      // trace(H_full) = 2 * sum(packed), recovering the reference class's curvature.
      double expected_trace = 0.0;
      for (auto v : p) {
        expected_trace += v * (1.0 - v);
      }
      EXPECT_NEAR(view.TotalCurvature(), weight * expected_trace, 1e-13 * std::max(1.0, weight))
          << "K=" << k;

      // The Hessian depends on the prediction only, never on the label.
      Stats other{p, (label + 1) % k, weight};
      for (std::size_t i = 0; i < d; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
          EXPECT_EQ(view.GetHessian(i, j), other.ConstView().GetHessian(i, j)) << "K=" << k;
        }
      }
    }
  }
}

TEST(ExactNumerics, FullHessianRowSumsVanishAndFreeBlockIsDefinite) {
  for (std::size_t k : {2ul, 3ul, 7ul}) {
    std::vector<double> z(k);
    for (std::size_t i = 0; i < k; ++i) {
      z[i] = 0.5 * static_cast<double>(i);
    }
    auto p = Softmax(z);

    // Every row of the full K x K Hessian sums to zero: that is the gauge direction the
    // K-1 parameterization removes.
    double worst_row = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
      double row = 0.0;
      for (std::size_t j = 0; j < k; ++j) {
        row += p[i] * ((i == j ? 1.0 : 0.0) - p[j]);
      }
      worst_row = std::max(worst_row, std::fabs(row));
    }
    EXPECT_LT(worst_row, 1e-15) << "K=" << k;

    // With the gauge removed and every p_i > 0, the free block is positive definite even
    // with no regularization at all.
    Stats stats{p, 0, 1.0};
    EXPECT_GT(SmallestEigenvalue(DenseSystem(stats.ConstView(), ExactL2::Raw(0.0)), k - 1), 0.0)
        << "K=" << k;
  }
}

// --------------------------------------------------------------------------- //
// Phase 7 - weight extremes
// --------------------------------------------------------------------------- //

TEST(ExactNumerics, WeightScalesStatisticsLinearly) {
  std::size_t constexpr kNumClasses = 4;
  auto d = kNumClasses - 1;
  std::vector<double> p{0.1, 0.2, 0.3, 0.4};

  Stats unit{p, 1, 1.0};
  auto base = unit.ConstView();
  for (double w : {0.0, 1e-12, 1e-8, 1.0, 1e2, 1e6}) {
    Stats scaled{p, 1, w};
    auto view = scaled.ConstView();
    for (std::size_t i = 0; i < d; ++i) {
      EXPECT_NEAR(view.GetGradient(i), w * base.GetGradient(i), 1e-12 * std::max(1.0, w))
          << "w=" << w;
      for (std::size_t j = 0; j <= i; ++j) {
        EXPECT_NEAR(view.GetHessian(i, j), w * base.GetHessian(i, j), 1e-12 * std::max(1.0, w))
            << "w=" << w;
      }
    }
    EXPECT_TRUE(std::isfinite(view.TotalCurvature())) << "w=" << w;
  }

  // A zero-weight row contributes nothing at all, so it cannot move a split...
  Stats zero{p, 1, 0.0};
  for (auto v : zero.ConstView().Data()) {
    EXPECT_EQ(v, 0.0);
  }
  // ...and a node made only of such rows is refused rather than solved, because H is then
  // exactly zero and the Newton step is undefined.
  ExactMultinomialLeafSolver solver{d};
  double gain = -1.0;
  EXPECT_FALSE(solver.Gain(zero.ConstView(), ExactL2::Raw(0.0), &gain));
  EXPECT_EQ(gain, -1.0);
}

// --------------------------------------------------------------------------- //
// Phase 6 - logit extremes
// --------------------------------------------------------------------------- //

TEST(ExactNumerics, ExtremeLogits) {
  std::size_t constexpr kNumClasses = 4;
  auto d = kNumClasses - 1;
  ExactMultinomialLeafSolver solver{d};

  for (double magnitude : {1.0, 10.0, 50.0, 100.0, 500.0}) {
    for (int sign : {-1, 1}) {
      auto s = static_cast<double>(sign);
      std::vector<double> z{magnitude * s, -magnitude * s, magnitude * s * 0.5, 0.0};
      auto p = Softmax(z);
      // The shifted softmax must still produce a distribution at |z| = 500, where a naive
      // exp() would overflow.
      ASSERT_NEAR(std::accumulate(p.cbegin(), p.cend(), 0.0), 1.0, 1e-12)
          << "magnitude=" << magnitude;
      for (auto v : p) {
        ASSERT_TRUE(std::isfinite(v));
        ASSERT_GE(v, 0.0);
      }

      Stats stats{p, 0, 1.0};
      for (double lambda : {0.0, 1.0}) {
        std::vector<double> w(d, 0.0);
        double gain = 0.0;
        if (!solver.Solve(stats.ConstView(), Centered(lambda, kNumClasses),
                          Span<double>{w.data(), w.size()}, &gain)) {
          continue;  // underflowed probabilities make H singular; refusal is correct
        }
        for (auto v : w) {
          EXPECT_TRUE(std::isfinite(v)) << "magnitude=" << magnitude;
        }
        EXPECT_TRUE(std::isfinite(gain)) << "magnitude=" << magnitude;
        EXPECT_GE(gain, -1e-9);
      }
    }
  }
}

// --------------------------------------------------------------------------- //
// Phase 15 - reference-class invariance, under stress
// --------------------------------------------------------------------------- //

TEST(ExactNumerics, ReferenceClassInvarianceUnderStress) {
  std::vector<std::vector<double>> cases{
      {0.25, 0.25, 0.25, 0.25},
      {0.97, 0.01, 0.01, 0.01},
      {1e-8, 0.3, 0.3, 0.4 - 1e-8},
  };
  std::size_t constexpr kLabel = 1;

  for (auto const& p : cases) {
    auto k = p.size();
    auto d = k - 1;
    ExactMultinomialLeafSolver solver{d};

    for (double lambda : {0.0, 1e-3, 1.0, 100.0}) {
      std::vector<std::vector<double>> centered_outputs;
      for (std::size_t reference = 0; reference < k; ++reference) {
        // The free coordinates are every class but the reference, in order.
        std::vector<std::size_t> order;
        for (std::size_t c = 0; c < k; ++c) {
          if (c != reference) {
            order.push_back(c);
          }
        }
        Stats stats{d};
        auto view = stats.View();
        for (std::size_t i = 0; i < d; ++i) {
          auto ci = order[i];
          view.AddGradient(i, p[ci] - (kLabel == ci ? 1.0 : 0.0));
          for (std::size_t j = 0; j <= i; ++j) {
            view.AddHessian(i, j, p[ci] * ((i == j ? 1.0 : 0.0) - p[order[j]]));
          }
        }

        std::vector<double> w(d, 0.0);
        if (!solver.Solve(stats.ConstView(), Centered(lambda, k),
                          Span<double>{w.data(), w.size()})) {
          centered_outputs.clear();
          break;  // refused for one reference class; nothing left to compare against
        }
        std::vector<double> full(k, 0.0);
        for (std::size_t i = 0; i < d; ++i) {
          full[order[i]] = w[i];
        }
        auto mean = std::accumulate(full.cbegin(), full.cend(), 0.0) / static_cast<double>(k);
        for (auto& v : full) {
          v -= mean;
        }
        centered_outputs.push_back(full);
      }
      if (centered_outputs.empty()) {
        continue;
      }
      ASSERT_EQ(centered_outputs.size(), k) << "lambda=" << lambda;
      for (std::size_t r = 1; r < k; ++r) {
        for (std::size_t c = 0; c < k; ++c) {
          EXPECT_NEAR(centered_outputs[r][c], centered_outputs[0][c], 1e-8)
              << "lambda=" << lambda << " reference " << r << " class " << c;
        }
      }
      // The centered output sums to zero, so repeated boosting cannot accumulate a drift
      // along the gauge direction.
      EXPECT_NEAR(std::accumulate(centered_outputs[0].cbegin(), centered_outputs[0].cend(), 0.0),
                  0.0, 1e-12);
    }
  }
}

TEST(ExactNumerics, RawGaugeIsNotReferenceClassInvariant) {
  // The counterpart to the test above: it records *why* the centered regularizer is the
  // default. With R = lambda I the fitted model depends on which class was pinned, so this
  // test failing would mean the two gauges had silently become the same thing.
  std::vector<double> p{0.6, 0.25, 0.1, 0.05};
  auto k = p.size();
  auto d = k - 1;
  ExactMultinomialLeafSolver solver{d};
  std::size_t constexpr kLabel = 1;
  double constexpr kLambda = 1.0;

  std::vector<std::vector<double>> outputs;
  for (std::size_t reference : {k - 1, std::size_t{0}}) {
    std::vector<std::size_t> order;
    for (std::size_t c = 0; c < k; ++c) {
      if (c != reference) {
        order.push_back(c);
      }
    }
    Stats stats{d};
    auto view = stats.View();
    for (std::size_t i = 0; i < d; ++i) {
      auto ci = order[i];
      view.AddGradient(i, p[ci] - (kLabel == ci ? 1.0 : 0.0));
      for (std::size_t j = 0; j <= i; ++j) {
        view.AddHessian(i, j, p[ci] * ((i == j ? 1.0 : 0.0) - p[order[j]]));
      }
    }
    std::vector<double> w(d, 0.0);
    ASSERT_TRUE(
        solver.Solve(stats.ConstView(), ExactL2::Raw(kLambda), Span<double>{w.data(), w.size()}));
    std::vector<double> full(k, 0.0);
    for (std::size_t i = 0; i < d; ++i) {
      full[order[i]] = w[i];
    }
    auto mean = std::accumulate(full.cbegin(), full.cend(), 0.0) / static_cast<double>(k);
    for (auto& v : full) {
      v -= mean;
    }
    outputs.push_back(full);
  }

  double worst = 0.0;
  for (std::size_t c = 0; c < k; ++c) {
    worst = std::max(worst, std::fabs(outputs[0][c] - outputs[1][c]));
  }
  std::printf("  raw gauge reference-class drift: %.3e\n", worst);
  EXPECT_GT(worst, 1e-3) << "the raw gauge has become reference-class invariant; if that is "
                            "intended, the centered regularizer is no longer load bearing";
}

// --------------------------------------------------------------------------- //
// Phase 16 - float transport against a double reference
// --------------------------------------------------------------------------- //

TEST(ExactNumerics, FloatTransportErrorIsQuantified) {
  std::mt19937 rng{99};
  std::uniform_real_distribution<double> logit{-5.0, 5.0};
  std::size_t constexpr kRows = 4096;

  for (std::size_t k : {3ul, 7ul, 10ul}) {
    auto d = k - 1;
    auto tri = PackedHessianSize(d);
    std::vector<double> exact_sum(tri, 0.0);
    std::vector<double> float_sum(tri, 0.0);

    for (std::size_t r = 0; r < kRows; ++r) {
      std::vector<double> z(k);
      for (auto& v : z) {
        v = logit(rng);
      }
      auto p = Softmax(z);
      double w = 0.5 + static_cast<double>(r % 4);
      std::size_t idx = 0;
      for (std::size_t i = 0; i < d; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
          auto value = w * p[i] * ((i == j ? 1.0 : 0.0) - p[j]);
          exact_sum[idx] += value;
          // Production stores the per-row sidecar as float and accumulates bins in double.
          float_sum[idx] += static_cast<double>(static_cast<float>(value));
          ++idx;
        }
      }
    }

    double worst_rel = 0.0;
    for (std::size_t i = 0; i < tri; ++i) {
      auto diff = std::fabs(exact_sum[i] - float_sum[i]);
      worst_rel = std::max(worst_rel, diff / std::max(1e-300, std::fabs(exact_sum[i])));
    }
    std::printf("  K=%zu float transport over %zu rows: rel %.3e\n", k, kRows, worst_rel);
    // Only the per-row rounding is lost; accumulating in double keeps the error at the
    // float epsilon of a single row rather than growing it with the row count.
    EXPECT_LT(worst_rel, 1e-6) << "K=" << k;
  }
}

// --------------------------------------------------------------------------- //
// Phase 17 - randomized property sweep, fixed seed
// --------------------------------------------------------------------------- //

TEST(ExactNumerics, RandomizedProperties) {
  std::mt19937 rng{20261122};  // fixed seed: the sweep is reproducible, not flaky
  std::uniform_real_distribution<double> logit{-8.0, 8.0};
  std::uniform_int_distribution<int> rows{1, 6};

  int checked = 0;
  for (std::size_t k : {2ul, 3ul, 4ul, 7ul}) {
    auto d = k - 1;
    ExactMultinomialLeafSolver solver{d};
    for (int trial = 0; trial < 120; ++trial) {
      Stats stats{d};
      auto n_rows = rows(rng);
      for (int r = 0; r < n_rows; ++r) {
        std::vector<double> z(k);
        for (auto& v : z) {
          v = logit(rng);
        }
        stats.Add(Softmax(z), static_cast<std::size_t>(r) % k, 0.1 + 0.3 * r);
      }
      auto view = stats.ConstView();

      for (std::size_t i = 0; i < d; ++i) {
        for (std::size_t j = 0; j < d; ++j) {
          ASSERT_EQ(view.GetHessian(i, j), view.GetHessian(j, i));
        }
      }
      // The trace identity is linear, so it survives accumulation over rows and weights.
      double packed_sum = 0.0;
      for (auto v : view.Hessian()) {
        packed_sum += v;
      }
      ASSERT_NEAR(view.TotalCurvature(), 2.0 * packed_sum, 1e-12 * std::max(1.0, packed_sum));

      auto reg = Centered(0.5, k);
      std::vector<double> w(d, 0.0);
      double solve_gain = 0.0;
      if (!solver.Solve(view, reg, Span<double>{w.data(), w.size()}, &solve_gain)) {
        continue;
      }
      double gain_only = 0.0;
      ASSERT_TRUE(solver.Gain(view, reg, &gain_only));
      auto scale = std::max(1.0, std::fabs(solve_gain));
      ASSERT_NEAR(gain_only, solve_gain, 1e-10 * scale);
      ASSERT_NEAR(solve_gain, -2.0 * DeltaL(view, reg, w), 1e-8 * scale);
      ASSERT_GE(solve_gain, -1e-12);

      auto a = DenseSystem(view, reg);
      ASSERT_LT(NewtonResidual(view, a, w), 1e-9 * std::max(1.0, MaxAbs(a) * MaxAbs(w)));
      ++checked;
    }
  }
  EXPECT_GT(checked, 300) << "too few systems survived for the sweep to mean anything";
}
}  // namespace xgboost::common
