/**
 * Copyright 2026, XGBoost Contributors
 */
#ifndef XGBOOST_TREE_HIST_EXACT_EVALUATOR_H_
#define XGBOOST_TREE_HIST_EXACT_EVALUATOR_H_

#include <cstddef>  // for size_t
#include <limits>   // for numeric_limits
#include <vector>   // for vector

#include "../../common/exact_multinomial/leaf_solver.h"    // for ExactMultinomialLeafSolver
#include "../../common/exact_multinomial/packed_stats.h"   // for PackedMultinomialStats
#include "../param.h"                                      // for TrainParam, IsValidSplit
#include "xgboost/base.h"                                  // for bst_target_t
#include "xgboost/logging.h"                               // for CHECK
#include "xgboost/span.h"                                  // for Span

namespace xgboost::tree {
/**
 * @brief Exact dense-Hessian split evaluation.
 *
 * XGBoost's scalar objective for a leaf output `w` is
 *
 *   dL(w) = G w + (1/2)(H + lambda) w^2 + alpha |w|
 *
 * and `CalcGainGivenWeight` returns `-(2 G w + (H + lambda) w^2 + 2 alpha |w|)`, i.e. gain
 * is `-2 dL` -- twice the loss reduction, not the loss reduction itself. Everything below
 * keeps that convention so that `min_split_loss` (gamma) and the parent/child comparison
 * retain their existing meaning.
 *
 * The dense analogue over the `K-1` free coordinates is
 *
 *   dL(w) = G^T w + (1/2) w^T (H + R) w,   (H + R) w* = -G,   gain = G^T (H + R)^-1 G
 *
 * with `R` from @ref common::ExactL2. For `K = 2` the free block is 1x1 and the gain
 * collapses to `G^2 / (H + lambda)`, exactly `tree::CalcGain`, which is the reduction check
 * the tests rely on.
 *
 * L1 (`reg_alpha`) and `max_delta_step` are deliberately absent: soft thresholding has no
 * closed form for a coupled system and a scalar step clip has no unique vector analogue.
 * Exact mode rejects both rather than silently applying a scalar rule (see @ref
 * CheckExactTrainParam).
 */

/** @brief Outcome of an exact leaf evaluation. */
struct ExactLeafGain {
  /** @brief `G^T (H + R)^-1 G`, XGBoost's gain convention. Zero when not evaluable. */
  double gain{0.0};
  /**
   * @brief Whether `H + R` was numerically positive definite.
   *
   * When false the leaf has no usable curvature in at least one coordinate. XGBoost's
   * scalar path treats that case as `weight 0, gain 0` (`param.h`, `sum_hess <= 0`), and
   * exact mode follows the same policy rather than inventing a damping rule.
   */
  bool valid{false};
};

/**
 * @brief Total curvature used for `min_child_weight`, normalized per class.
 *
 * Exact mode compares `min_child_weight` against `trace(H_full) / K`, matching the SHAPE of
 * the existing vector-leaf gate (`split_evaluator.h`, `IsValidSplit(param, left_hess / k,
 * right_hess / k)`), which `param.h` documents as "the normalized Hessian trace".
 *
 * It is NOT the same NUMBER, and that difference is deliberate. The diagonal multiclass
 * objective does not store `p_k (1 - p_k)`; it stores the absolute-residual pseudo-Hessian
 * `|p_k - y_k| * w` (see `MulticlassClassGradient` in `multiclass_obj.cc`). Summed over
 * classes and divided by `K`, the two gates differ:
 *
 *   diagonal:  sum_k |p_k - y_k| * w / K
 *   exact:     w * (1 - sum_k p_k^2) / K
 *
 * At a uniform prediction the diagonal gate is exactly 2x the exact one for every `K`;
 * as predictions sharpen the ratio falls towards 1. End to end this shows up as roughly
 * 4-10% fewer nodes in exact mode at the same `min_child_weight`.
 *
 * Exact mode uses true curvature because that is the quantity its Newton system actually
 * inverts: gating on the pseudo-Hessian would let a node through whose real curvature cannot
 * support a leaf value. Rescaling either side to force numeric agreement would make the gate
 * mean something different from the matrix being factorized, so the semantics are left
 * honest and the difference is documented (see also `multi_hessian` in the parameter docs).
 *
 * Two properties come for free:
 *
 *  - the trace is taken over all `K` classes, not the `K-1` stored ones, so the value does
 *    not depend on which class happens to be the reference;
 *  - it is recovered from the stored free block exactly, via
 *    `trace(H_full) = 2 * sum(packed triangle)` (see
 *    @ref common::PackedHessianView::TotalCurvature), so nothing extra is stored.
 *
 * The result equals `sum_rows w (1 - sum_k p_k^2) / K`, the weighted Gini impurity of the
 * predicted distribution per class.
 */
[[nodiscard]] inline double ExactChildCurvature(
    common::PackedMultinomialStats<double const> stats, bst_target_t n_classes) {
  CHECK_GT(n_classes, 0);
  return stats.TotalCurvature() / static_cast<double>(n_classes);
}

/**
 * @brief Evaluate one node's exact gain and leaf weight.
 *
 * @param solver Reused across calls; holds the factorization scratch.
 * @param out_w  Optional leaf weight output, `K-1` entries.
 */
[[nodiscard]] inline ExactLeafGain EvaluateExactLeaf(
    common::ExactMultinomialLeafSolver* solver, common::PackedMultinomialStats<double const> stats,
    common::ExactL2 reg, common::Span<double> out_w) {
  ExactLeafGain out;
  double gain = 0.0;
  if (solver->Solve(stats, reg, out_w, &gain)) {
    out.gain = gain;
    out.valid = true;
  }
  return out;
}

/** @brief Gain only, for callers that do not need the weight. */
[[nodiscard]] inline ExactLeafGain EvaluateExactGain(
    common::ExactMultinomialLeafSolver* solver, common::PackedMultinomialStats<double const> stats,
    common::ExactL2 reg) {
  ExactLeafGain out;
  double gain = 0.0;
  if (solver->Gain(stats, reg, &gain)) {
    out.gain = gain;
    out.valid = true;
  }
  return out;
}

/**
 * @brief Loss change of a candidate split, in XGBoost's gain units.
 *
 * `loss_chg = gain(left) + gain(right) - gain(parent)`, matching the scalar evaluator, so
 * it is directly comparable against `min_split_loss`.
 *
 * @p parent_gain is supplied by the caller rather than recomputed. The parent statistics are
 * constant across every candidate split of a node, so factorizing them per bin repeats an
 * O(d^3) solve for a value that cannot change. Callers enumerating a node compute it once;
 * see the convenience overload below for one-off use.
 *
 * Returns negative infinity -- the repository's marker for an unusable candidate -- when
 * either child fails `min_child_weight` or when either child's regularized Hessian is not
 * positive definite. A child with no curvature cannot be given a defensible leaf value, so
 * admitting the split would commit the tree to a weight the solve did not produce.
 */
[[nodiscard]] inline double ExactSplitLossChange(
    common::ExactMultinomialLeafSolver* solver, TrainParam const& param, bst_target_t n_classes,
    double parent_gain, common::PackedMultinomialStats<double const> left,
    common::PackedMultinomialStats<double const> right) {
  constexpr auto kNegInf = -std::numeric_limits<double>::infinity();

  auto left_curvature = ExactChildCurvature(left, n_classes);
  auto right_curvature = ExactChildCurvature(right, n_classes);
  if (!IsValidSplit(param, left_curvature, right_curvature)) {
    return kNegInf;
  }

  auto reg = common::ExactL2::Centered(param.reg_lambda, n_classes);
  auto left_gain = EvaluateExactGain(solver, left, reg);
  if (!left_gain.valid) {
    return kNegInf;
  }
  auto right_gain = EvaluateExactGain(solver, right, reg);
  if (!right_gain.valid) {
    return kNegInf;
  }
  return left_gain.gain + right_gain.gain - parent_gain;
}

/**
 * @brief The gain of a node, used as the baseline its splits are measured against.
 *
 * A parent whose own system is singular still has a well defined children sum; reporting
 * zero for it matches the scalar path, where `CalcGain` returns 0 for `sum_hess <= 0`.
 * @ref ExactLeafGain already carries zero in that case.
 */
[[nodiscard]] inline double ExactParentGain(common::ExactMultinomialLeafSolver* solver,
                                            TrainParam const& param, bst_target_t n_classes,
                                            common::PackedMultinomialStats<double const> parent) {
  auto reg = common::ExactL2::Centered(param.reg_lambda, n_classes);
  return EvaluateExactGain(solver, parent, reg).gain;
}

/**
 * @brief Convenience overload computing the parent gain itself.
 *
 * Correct but wasteful inside an enumeration loop; prefer hoisting @ref ExactParentGain.
 */
[[nodiscard]] inline double ExactSplitLossChange(
    common::ExactMultinomialLeafSolver* solver, TrainParam const& param, bst_target_t n_classes,
    common::PackedMultinomialStats<double const> parent,
    common::PackedMultinomialStats<double const> left,
    common::PackedMultinomialStats<double const> right) {
  auto parent_gain = ExactParentGain(solver, param, n_classes, parent);
  return ExactSplitLossChange(solver, param, n_classes, parent_gain, left, right);
}

/**
 * @brief Reject training parameters that exact mode cannot honour.
 *
 * Every rejection here is a configuration that would otherwise silently fall back to a
 * scalar-Hessian rule and produce a mathematically different model.
 */
inline void CheckExactTrainParam(TrainParam const& param) {
  CHECK_EQ(param.reg_alpha, 0.0f)
      << "reg_alpha is not supported with multi_hessian=exact: L1 soft thresholding has no "
         "closed form for a coupled Newton system, where the K-1 outputs cannot be "
         "thresholded independently. Set reg_alpha=0, or use multi_hessian=diagonal.";
  CHECK_EQ(param.max_delta_step, 0.0f)
      << "max_delta_step is not supported with multi_hessian=exact: it clips a scalar leaf "
         "value, and an exact leaf is a vector of K coupled outputs with no unique quantity "
         "to clip. Set max_delta_step=0, or use multi_hessian=diagonal.";
  CHECK(!param.HasMonotone())
      << "monotone_constraints are not supported with multi_hessian=exact. The "
         "constraint bounds a scalar leaf value against its sibling, but an exact leaf is a "
         "vector of K coupled class outputs, so there is no unique quantity to constrain. Use "
         "multi_hessian=diagonal, or drop the constraint.";
}
}  // namespace xgboost::tree
#endif  // XGBOOST_TREE_HIST_EXACT_EVALUATOR_H_
