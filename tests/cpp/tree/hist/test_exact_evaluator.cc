/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>

#include <algorithm>  // for find
#include <cmath>    // for fabs
#include <cstdio>   // for printf
#include <cstddef>  // for size_t
#include <vector>   // for vector

#include "../../../../src/common/exact_multinomial/leaf_solver.h"
#include "../../../../src/common/exact_multinomial/packed_stats.h"
#include "../../../../src/tree/hist/exact_evaluator.h"
#include "../../../../src/tree/param.h"

namespace xgboost::tree {
namespace {
/** @brief A packed statistic owning its storage, for tests. */
class Stats {
 public:
  explicit Stats(bst_target_t n_free)
      : n_free_{n_free}, buffer_(common::PackedStatsStride(n_free), 0.0) {}

  [[nodiscard]] common::PackedMultinomialStats<double> View() {
    return {common::Span<double>{buffer_.data(), buffer_.size()}, n_free_};
  }
  [[nodiscard]] common::PackedMultinomialStats<double const> ConstView() const {
    return {common::Span<double const>{buffer_.data(), buffer_.size()}, n_free_};
  }

  /** @brief Accumulate one row with probabilities @p p, label @p label and weight @p w. */
  void AddRow(std::vector<double> const& p, std::size_t label, double w) {
    auto stats = this->View();
    for (std::size_t i = 0; i < n_free_; ++i) {
      stats.AddGradient(i, w * (p[i] - (label == i ? 1.0 : 0.0)));
      for (std::size_t j = 0; j <= i; ++j) {
        stats.AddHessian(i, j, w * p[i] * ((i == j ? 1.0 : 0.0) - p[j]));
      }
    }
  }

 private:
  bst_target_t n_free_;
  std::vector<double> buffer_;
};

TrainParam MakeParam(double lambda, double min_child_weight = 0.0, double gamma = 0.0) {
  TrainParam param;
  param.UpdateAllowUnknown(Args{{"reg_lambda", std::to_string(lambda)},
                                {"min_child_weight", std::to_string(min_child_weight)},
                                {"min_split_loss", std::to_string(gamma)}});
  return param;
}
}  // anonymous namespace

/** The regularization matrix R has the derived structure and stays positive definite. */
TEST(ExactEvaluator, CenteredRegularizerIsPositiveDefinite) {
  double constexpr kLambda = 0.5;
  bst_target_t constexpr kNumClasses = 4;
  auto centered = common::ExactL2::Centered(kLambda, kNumClasses);
  auto raw = common::ExactL2::Raw(kLambda);

  EXPECT_TRUE(centered.IsCentered());
  EXPECT_FALSE(raw.IsCentered());
  // R = lambda (I - 11^T/K).
  EXPECT_DOUBLE_EQ(centered.Diagonal(), kLambda * (1.0 - 1.0 / kNumClasses));
  EXPECT_DOUBLE_EQ(centered.OffDiagonal(), -kLambda / kNumClasses);
  EXPECT_DOUBLE_EQ(raw.Diagonal(), kLambda);
  EXPECT_DOUBLE_EQ(raw.OffDiagonal(), 0.0);

  // Eigenvalues of (I - 11^T/K) over K-1 coordinates are 1 (multiplicity K-2) and 1/K, so R
  // is positive definite. Verify the smallest directly: the all-ones direction.
  auto d = static_cast<std::size_t>(kNumClasses - 1);
  std::vector<double> ones(d, 1.0);
  double quad = 0.0;
  for (std::size_t i = 0; i < d; ++i) {
    for (std::size_t j = 0; j < d; ++j) {
      quad += ones[i] * ones[j] * (i == j ? centered.Diagonal() : centered.OffDiagonal());
    }
  }
  // 1^T R 1 = lambda * d * (1 - d/K) = lambda * d / K for d = K-1.
  EXPECT_NEAR(quad, kLambda * static_cast<double>(d) / kNumClasses, 1e-12);
  EXPECT_GT(quad, 0.0);
}

/**
 * K=2 must reduce exactly to the scalar path. This is the strongest available cross-check:
 * the dense gain is compared against XGBoost's own CalcGain on the same statistics.
 */
TEST(ExactEvaluator, ReducesToScalarGainForTwoClasses) {
  bst_target_t constexpr kNumFree = 1;
  common::ExactMultinomialLeafSolver solver{kNumFree};

  for (double lambda : {0.0, 0.5, 2.0}) {
    auto param = MakeParam(lambda);
    for (double p0 : {0.1, 0.5, 0.9}) {
      for (std::size_t label : {0ul, 1ul}) {
        Stats stats{kNumFree};
        stats.AddRow({p0, 1.0 - p0}, label, 1.0);

        auto view = stats.ConstView();
        auto g = view.GetGradient(0);
        auto h = view.GetHessian(0, 0);

        // Raw gauge is lambda*I, which for one free coordinate is the scalar case.
        auto out = EvaluateExactGain(&solver, view, common::ExactL2::Raw(lambda));
        ASSERT_TRUE(out.valid) << "p0=" << p0 << " lambda=" << lambda;

        auto expected = tree::CalcGain(param, g, h);
        EXPECT_NEAR(out.gain, expected, 1e-12)
            << "p0=" << p0 << " label=" << label << " lambda=" << lambda;

        // And the weight matches CalcWeight.
        std::vector<double> w(1, 0.0);
        ASSERT_TRUE(solver.Solve(view, common::ExactL2::Raw(lambda),
                                 common::Span<double>{w.data(), w.size()}));
        EXPECT_NEAR(w[0], tree::CalcWeight(param, g, h), 1e-12);
      }
    }
  }
}

/** The gain equals -2 * dL(w*), computed independently from the Taylor objective. */
TEST(ExactEvaluator, GainMatchesTaylorObjective) {
  bst_target_t constexpr kNumClasses = 4;
  bst_target_t constexpr kNumFree = kNumClasses - 1;
  double constexpr kLambda = 0.3;

  Stats stats{kNumFree};
  stats.AddRow({0.1, 0.2, 0.3, 0.4}, 0, 1.0);
  stats.AddRow({0.25, 0.25, 0.25, 0.25}, 2, 2.0);
  stats.AddRow({0.05, 0.6, 0.15, 0.2}, 3, 0.5);
  auto view = stats.ConstView();

  common::ExactMultinomialLeafSolver solver{kNumFree};
  auto reg = common::ExactL2::Centered(kLambda, kNumClasses);
  std::vector<double> w(kNumFree, 0.0);
  double gain = 0.0;
  ASSERT_TRUE(solver.Solve(view, reg, common::Span<double>{w.data(), w.size()}, &gain));

  // Independent evaluation of dL(w) = G^T w + 0.5 w^T (H + R) w.
  auto objective = [&](std::vector<double> const& x) {
    double linear = 0.0;
    for (std::size_t i = 0; i < kNumFree; ++i) {
      linear += view.GetGradient(i) * x[i];
    }
    double quad = 0.0;
    for (std::size_t i = 0; i < kNumFree; ++i) {
      for (std::size_t j = 0; j < kNumFree; ++j) {
        auto a = view.GetHessian(i, j) + (i == j ? reg.Diagonal() : reg.OffDiagonal());
        quad += x[i] * a * x[j];
      }
    }
    return linear + 0.5 * quad;
  };

  auto delta = objective(w);
  EXPECT_NEAR(gain, -2.0 * delta, 1e-10);
  EXPECT_GT(gain, 0.0);

  // w* is a strict minimum: perturbing any coordinate increases the objective.
  for (std::size_t i = 0; i < kNumFree; ++i) {
    for (double step : {-0.05, 0.05}) {
      auto perturbed = w;
      perturbed[i] += step;
      EXPECT_GT(objective(perturbed), delta) << "coordinate " << i << " step " << step;
    }
  }
}

/** Gain is non-negative and the split loss change reconciles parent and children. */
TEST(ExactEvaluator, SplitLossChangeReconcilesParentAndChildren) {
  bst_target_t constexpr kNumClasses = 3;
  bst_target_t constexpr kNumFree = kNumClasses - 1;
  auto param = MakeParam(0.5);
  common::ExactMultinomialLeafSolver solver{kNumFree};

  Stats left{kNumFree}, right{kNumFree}, parent{kNumFree};
  left.AddRow({0.1, 0.3, 0.6}, 0, 1.0);
  left.AddRow({0.2, 0.2, 0.6}, 1, 1.5);
  right.AddRow({0.7, 0.2, 0.1}, 2, 2.0);
  right.AddRow({0.4, 0.4, 0.2}, 0, 1.0);
  // Parent is the sum of the children, as the histogram guarantees.
  parent.AddRow({0.1, 0.3, 0.6}, 0, 1.0);
  parent.AddRow({0.2, 0.2, 0.6}, 1, 1.5);
  parent.AddRow({0.7, 0.2, 0.1}, 2, 2.0);
  parent.AddRow({0.4, 0.4, 0.2}, 0, 1.0);

  auto reg = common::ExactL2::Centered(param.reg_lambda, kNumClasses);
  auto gain_left = EvaluateExactGain(&solver, left.ConstView(), reg);
  auto gain_right = EvaluateExactGain(&solver, right.ConstView(), reg);
  auto gain_parent = EvaluateExactGain(&solver, parent.ConstView(), reg);
  ASSERT_TRUE(gain_left.valid);
  ASSERT_TRUE(gain_right.valid);
  ASSERT_TRUE(gain_parent.valid);

  // G^T A^-1 G is a positive definite quadratic form, so every gain is non-negative.
  EXPECT_GE(gain_left.gain, 0.0);
  EXPECT_GE(gain_right.gain, 0.0);
  EXPECT_GE(gain_parent.gain, 0.0);

  auto loss_chg = ExactSplitLossChange(&solver, param, kNumClasses, parent.ConstView(),
                                       left.ConstView(), right.ConstView());
  EXPECT_NEAR(loss_chg, gain_left.gain + gain_right.gain - gain_parent.gain, 1e-12);

  // Splitting into genuinely different children must help under this convention.
  EXPECT_GT(loss_chg, 0.0);
}

/** A split whose children are statistically identical to the parent gains nothing. */
TEST(ExactEvaluator, NoGainFromUninformativeSplit) {
  bst_target_t constexpr kNumClasses = 3;
  bst_target_t constexpr kNumFree = kNumClasses - 1;
  auto param = MakeParam(0.0);
  common::ExactMultinomialLeafSolver solver{kNumFree};

  // Two halves with identical statistics: the parent's optimum is already each child's.
  Stats left{kNumFree}, right{kNumFree}, parent{kNumFree};
  for (auto* s : {&left, &right}) {
    s->AddRow({0.2, 0.3, 0.5}, 1, 1.0);
  }
  parent.AddRow({0.2, 0.3, 0.5}, 1, 1.0);
  parent.AddRow({0.2, 0.3, 0.5}, 1, 1.0);

  auto loss_chg = ExactSplitLossChange(&solver, param, kNumClasses, parent.ConstView(),
                                       left.ConstView(), right.ConstView());
  // gain is homogeneous of degree 1 in the statistics when lambda = 0:
  // 2 * gain(S) - gain(2S) = 2g - 2g = 0.
  EXPECT_NEAR(loss_chg, 0.0, 1e-9);
}

/** min_child_weight uses the reference-invariant per-class curvature. */
TEST(ExactEvaluator, MinChildWeightRejectsThinChildren) {
  bst_target_t constexpr kNumClasses = 3;
  bst_target_t constexpr kNumFree = kNumClasses - 1;
  common::ExactMultinomialLeafSolver solver{kNumFree};

  Stats left{kNumFree}, right{kNumFree}, parent{kNumFree};
  std::vector<double> p{0.2, 0.3, 0.5};
  left.AddRow(p, 0, 0.01);  // deliberately tiny weight
  right.AddRow(p, 1, 5.0);
  parent.AddRow(p, 0, 0.01);
  parent.AddRow(p, 1, 5.0);

  // Curvature per class = w * (1 - sum p^2) / K.
  double gini = 1.0;
  for (auto v : p) {
    gini -= v * v;
  }
  auto left_curv = ExactChildCurvature(left.ConstView(), kNumClasses);
  EXPECT_NEAR(left_curv, 0.01 * gini / kNumClasses, 1e-12);

  // Threshold below the thin child: accepted.
  auto lenient = MakeParam(0.1, left_curv * 0.5);
  EXPECT_GT(ExactSplitLossChange(&solver, lenient, kNumClasses, parent.ConstView(),
                                 left.ConstView(), right.ConstView()),
            -std::numeric_limits<double>::infinity());

  // Threshold above the thin child: rejected outright.
  auto strict = MakeParam(0.1, left_curv * 2.0);
  EXPECT_EQ(ExactSplitLossChange(&solver, strict, kNumClasses, parent.ConstView(),
                                 left.ConstView(), right.ConstView()),
            -std::numeric_limits<double>::infinity());
}

/**
 * Degenerate curvature must follow the repository's policy: no gain, no weight, and never a
 * NaN or an arbitrary step.
 */
TEST(ExactEvaluator, SingularHessianIsRejected) {
  bst_target_t constexpr kNumClasses = 3;
  bst_target_t constexpr kNumFree = kNumClasses - 1;
  common::ExactMultinomialLeafSolver solver{kNumFree};

  // A class with zero probability leaves a zero row and column.
  Stats degenerate{kNumFree};
  degenerate.AddRow({0.0, 0.4, 0.6}, 1, 1.0);

  auto unregularized =
      EvaluateExactGain(&solver, degenerate.ConstView(), common::ExactL2::Raw(0.0));
  EXPECT_FALSE(unregularized.valid);
  EXPECT_EQ(unregularized.gain, 0.0);

  // Regularization restores definiteness, and the gain is then finite and non-negative.
  auto regularized = EvaluateExactGain(
      &solver, degenerate.ConstView(), common::ExactL2::Centered(0.5, kNumClasses));
  EXPECT_TRUE(regularized.valid);
  EXPECT_TRUE(std::isfinite(regularized.gain));
  EXPECT_GE(regularized.gain, 0.0);

  // An unusable child kills the candidate rather than producing a number.
  Stats healthy{kNumFree};
  healthy.AddRow({0.3, 0.3, 0.4}, 0, 1.0);
  auto param = MakeParam(0.0);
  auto loss_chg = ExactSplitLossChange(&solver, param, kNumClasses, healthy.ConstView(),
                                       degenerate.ConstView(), healthy.ConstView());
  EXPECT_EQ(loss_chg, -std::numeric_limits<double>::infinity());
}

/**
 * The centered gauge makes the model invariant to the choice of reference class; the raw
 * gauge does not. This is the property that decided the regularization design.
 */
TEST(ExactEvaluator, CenteredGaugeIsReferenceClassInvariant) {
  std::vector<double> p{0.1, 0.2, 0.3, 0.4};
  auto n_classes = static_cast<bst_target_t>(p.size());
  auto n_free = static_cast<bst_target_t>(n_classes - 1);
  std::size_t constexpr kLabel = 1;
  double constexpr kLambda = 0.7;

  // For each choice of reference class, solve and map the K-1 weights back to the centered
  // K-output representation. The resulting K-vector must be identical.
  auto solve_with_reference = [&](std::size_t reference) {
    std::vector<std::size_t> order;  // free classes in order, then the reference
    for (std::size_t k = 0; k < p.size(); ++k) {
      if (k != reference) {
        order.push_back(k);
      }
    }

    std::vector<double> buffer(common::PackedStatsStride(n_free), 0.0);
    auto stats = common::PackedMultinomialStats<double>{
        common::Span<double>{buffer.data(), buffer.size()}, n_free};
    for (std::size_t i = 0; i < n_free; ++i) {
      auto ci = order[i];
      stats.AddGradient(i, p[ci] - (kLabel == ci ? 1.0 : 0.0));
      for (std::size_t j = 0; j <= i; ++j) {
        auto cj = order[j];
        stats.AddHessian(i, j, p[ci] * ((i == j ? 1.0 : 0.0) - p[cj]));
      }
    }

    common::ExactMultinomialLeafSolver solver{n_free};
    std::vector<double> w(n_free, 0.0);
    EXPECT_TRUE(solver.Solve(
        common::PackedMultinomialStats<double const>{
            common::Span<double const>{buffer.data(), buffer.size()}, n_free},
        common::ExactL2::Centered(kLambda, n_classes),
        common::Span<double>{w.data(), w.size()}));

    // Embed into K outputs (reference gets 0), then center.
    std::vector<double> full(p.size(), 0.0);
    for (std::size_t i = 0; i < n_free; ++i) {
      full[order[i]] = w[i];
    }
    double mean = 0.0;
    for (auto v : full) {
      mean += v;
    }
    mean /= static_cast<double>(full.size());
    for (auto& v : full) {
      v -= mean;
    }
    return full;
  };

  auto reference_zero = solve_with_reference(0);
  for (std::size_t reference = 1; reference < p.size(); ++reference) {
    auto other = solve_with_reference(reference);
    ASSERT_EQ(other.size(), reference_zero.size());
    for (std::size_t k = 0; k < other.size(); ++k) {
      EXPECT_NEAR(other[k], reference_zero[k], 1e-10)
          << "reference class " << reference << ", output " << k;
    }
  }

  // The centered outputs sum to zero by construction, so no gauge drift accumulates.
  double total = 0.0;
  for (auto v : reference_zero) {
    total += v;
  }
  EXPECT_NEAR(total, 0.0, 1e-12);
}

/**
 * The parent gain does not depend on the candidate split, which is what makes hoisting it
 * out of the enumeration loop a pure optimization.
 *
 * Two things are asserted: the parent gain is invariant across wildly different partitions
 * of the same parent, and the hoisted form of ExactSplitLossChange agrees bit-for-bit with
 * the form that recomputes it. A regression in either would silently change split choices.
 */
TEST(ExactEvaluator, ParentGainIsIndependentOfCandidateSplit) {
  bst_target_t constexpr kNumClasses = 4;
  bst_target_t constexpr kNumFree = kNumClasses - 1;
  auto param = MakeParam(0.7);
  common::ExactMultinomialLeafSolver solver{kNumFree};

  // A parent built from several rows, then split three different ways.
  std::vector<std::vector<double>> rows{
      {0.1, 0.2, 0.3, 0.4}, {0.4, 0.3, 0.2, 0.1}, {0.25, 0.25, 0.25, 0.25}, {0.05, 0.6, 0.15, 0.2}};
  std::vector<std::size_t> labels{0, 1, 2, 3};
  std::vector<double> weights{1.0, 2.0, 0.5, 1.5};

  Stats parent{kNumFree};
  for (std::size_t r = 0; r < rows.size(); ++r) {
    parent.AddRow(rows[r], labels[r], weights[r]);
  }

  auto expected_parent_gain =
      ExactParentGain(&solver, param, kNumClasses, parent.ConstView());
  EXPECT_GT(expected_parent_gain, 0.0);

  // Every partition of the same rows must see the same parent gain, and the hoisted and
  // recomputing forms must agree exactly.
  std::vector<std::vector<std::size_t>> partitions{{0}, {0, 1}, {0, 1, 2}, {1, 3}};
  for (auto const& left_rows : partitions) {
    Stats left{kNumFree};
    Stats right{kNumFree};
    for (std::size_t r = 0; r < rows.size(); ++r) {
      auto in_left =
          std::find(left_rows.cbegin(), left_rows.cend(), r) != left_rows.cend();
      (in_left ? left : right).AddRow(rows[r], labels[r], weights[r]);
    }

    // Recomputed-parent form (the pre-optimization behaviour).
    auto recomputed = ExactSplitLossChange(&solver, param, kNumClasses, parent.ConstView(),
                                           left.ConstView(), right.ConstView());
    // Hoisted-parent form (what the enumerator now uses).
    auto hoisted = ExactSplitLossChange(&solver, param, kNumClasses, expected_parent_gain,
                                        left.ConstView(), right.ConstView());
    EXPECT_DOUBLE_EQ(recomputed, hoisted)
        << "hoisting the parent gain changed the loss change";

    // And recomputing the parent after the children were solved still gives the same value,
    // i.e. the solver carries no state between calls that would perturb it.
    auto reparented = ExactParentGain(&solver, param, kNumClasses, parent.ConstView());
    EXPECT_DOUBLE_EQ(reparented, expected_parent_gain)
        << "the parent gain moved after solving the children";
  }

  // The parent statistics themselves must be untouched by all of the above.
  auto after = ExactParentGain(&solver, param, kNumClasses, parent.ConstView());
  EXPECT_DOUBLE_EQ(after, expected_parent_gain);
}

/** Exact mode rejects parameters it cannot honour instead of approximating them. */
TEST(ExactEvaluator, RejectsUnsupportedParameters) {
  auto ok = MakeParam(1.0);
  EXPECT_NO_THROW(CheckExactTrainParam(ok));

  TrainParam alpha;
  alpha.UpdateAllowUnknown(Args{{"reg_alpha", "0.5"}});
  EXPECT_THROW(CheckExactTrainParam(alpha), dmlc::Error);

  TrainParam delta;
  delta.UpdateAllowUnknown(Args{{"max_delta_step", "1.0"}});
  EXPECT_THROW(CheckExactTrainParam(delta), dmlc::Error);
}

/**
 * `min_child_weight` gates on a DIFFERENT quantity in the two modes, deliberately.
 *
 * `diagonal` multiclass stores the absolute-residual pseudo-Hessian `|p_k - y_k| * w`
 * (`MulticlassClassGradient` in multiclass_obj.cc); `exact` stores true curvature
 * `p_i (delta_ij - p_j) * w`. Both are then divided by `K`, matching
 * `split_evaluator.h`'s `IsValidSplit(param, left_hess / k, right_hess / k)`.
 *
 * This test pins the relationship the parameter documentation states, so the code and the
 * docs cannot drift apart silently. It is NOT asserting that the two agree -- it asserts
 * precisely how they differ.
 */
TEST(ExactEvaluator, MinChildWeightGateDiffersFromDiagonalByDesign) {
  auto diagonal_gate = [](std::vector<double> const& p, std::size_t label, double w) {
    // What the diagonal multiclass path accumulates, summed over classes and divided by K.
    double total = 0.0;
    for (std::size_t k = 0; k < p.size(); ++k) {
      total += std::max(std::fabs(p[k] - (label == k ? 1.0 : 0.0)) * w, 1e-16);
    }
    return total / static_cast<double>(p.size());
  };

  for (bst_target_t k : {3u, 5u, 7u, 10u}) {
    auto n_free = static_cast<bst_target_t>(k - 1);
    // Uniform prediction: the documented worst case, where the ratio is exactly 2.
    std::vector<double> uniform(k, 1.0 / static_cast<double>(k));
    Stats stats{n_free};
    auto view = stats.View();
    for (std::size_t i = 0; i < n_free; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        view.AddHessian(i, j, uniform[i] * ((i == j ? 1.0 : 0.0) - uniform[j]));
      }
    }
    auto exact = ExactChildCurvature(stats.ConstView(), k);
    auto diagonal = diagonal_gate(uniform, 0, 1.0);
    EXPECT_NEAR(diagonal / exact, 2.0, 1e-9)
        << "K=" << k << ": the documented 2x ratio at a uniform prediction no longer holds";

    // Sharp prediction: the ratio falls towards 1, as documented.
    std::vector<double> sharp(k, 0.01 / static_cast<double>(k - 1));
    sharp[0] = 0.99;
    Stats sharp_stats{n_free};
    auto sview = sharp_stats.View();
    for (std::size_t i = 0; i < n_free; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        sview.AddHessian(i, j, sharp[i] * ((i == j ? 1.0 : 0.0) - sharp[j]));
      }
    }
    auto sharp_ratio =
        diagonal_gate(sharp, 0, 1.0) / ExactChildCurvature(sharp_stats.ConstView(), k);
    EXPECT_LT(sharp_ratio, 1.2) << "K=" << k << ": ratio should approach 1 as p sharpens";
    EXPECT_GE(sharp_ratio, 1.0) << "K=" << k;
    std::printf("  K=%u  min_child_weight gate ratio diagonal/exact: uniform=%.3f sharp=%.3f\n",
                k, diagonal / exact, sharp_ratio);
  }

  // The exact gate is reference-class invariant, which the diagonal one cannot be because it
  // depends on the label. This is the property that makes it usable as a split constraint.
  std::vector<double> p{0.5, 0.3, 0.2};
  Stats a{2};
  Stats b{2};
  auto fill = [&](Stats* into, std::vector<std::size_t> const& order) {
    auto v = into->View();
    for (std::size_t i = 0; i < 2; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        v.AddHessian(i, j, p[order[i]] * ((i == j ? 1.0 : 0.0) - p[order[j]]));
      }
    }
  };
  fill(&a, {0, 1});  // reference class 2
  fill(&b, {1, 2});  // reference class 0
  EXPECT_NEAR(ExactChildCurvature(a.ConstView(), 3), ExactChildCurvature(b.ConstView(), 3),
              1e-15)
      << "the exact min_child_weight gate depends on the reference class";
}
}  // namespace xgboost::tree
