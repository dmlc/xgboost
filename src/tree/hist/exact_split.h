/**
 * Copyright 2026, XGBoost Contributors
 */
#ifndef XGBOOST_TREE_HIST_EXACT_SPLIT_H_
#define XGBOOST_TREE_HIST_EXACT_SPLIT_H_

#include <algorithm>  // for max
#include <cstddef>    // for size_t
#include <limits>     // for numeric_limits
#include <vector>     // for vector

#include "../../common/exact_multinomial/leaf_solver.h"   // for ExactMultinomialLeafSolver
#include "../../common/exact_multinomial/packed_stats.h"  // for PackedMultinomialStats
#include "../../common/hist_util.h"                       // for HistogramCuts
#include "../param.h"                                     // for TrainParam
#include "exact_evaluator.h"                              // for ExactSplitLossChange
#include "exact_histogram.h"                              // for ExactHistRecordSize
#include "xgboost/base.h"                                 // for bst_feature_t
#include "xgboost/span.h"                                 // for Span

namespace xgboost::tree {
/**
 * @brief Best split found for one node under the exact dense-Hessian objective.
 */
struct ExactSplitCandidate {
  double loss_chg{-std::numeric_limits<double>::infinity()};
  bst_feature_t fidx{0};
  float split_value{0.0f};
  /** @brief Whether rows with a missing value go to the left child. */
  bool default_left{false};
  bool valid{false};

  /** @brief Adopt @p that if it is a strict improvement. */
  bool Update(ExactSplitCandidate const& that) {
    if (that.valid && that.loss_chg > this->loss_chg) {
      *this = that;
      return true;
    }
    return false;
  }
};

/**
 * @brief Owning scratch for one node's packed statistics.
 *
 * Split enumeration needs running `left` and derived `right` records. Allocating them once
 * per node and reusing them across every bin keeps the inner loop allocation free.
 */
class ExactStatBuffer {
 public:
  void Reset(bst_target_t n_free) {
    n_free_ = n_free;
    buffer_.assign(common::PackedStatsStride(n_free), 0.0);
  }
  void Zero() { std::fill(buffer_.begin(), buffer_.end(), 0.0); }

  [[nodiscard]] common::PackedMultinomialStats<double> View() {
    return {common::Span<double>{buffer_.data(), buffer_.size()}, n_free_};
  }
  [[nodiscard]] common::PackedMultinomialStats<double const> ConstView() const {
    return {common::Span<double const>{buffer_.data(), buffer_.size()}, n_free_};
  }
  [[nodiscard]] common::Span<double> Data() {
    return common::Span<double>{buffer_.data(), buffer_.size()};
  }

 private:
  bst_target_t n_free_{0};
  std::vector<double> buffer_;
};

/**
 * @brief Read a node's total statistics -- the sum over its rows, each counted once.
 *
 * This is NOT the sum of the bins. Every row lands in one bin per feature, so summing bins
 * across all features counts each row `n_features` times, and summing one feature's bins
 * misses the rows for which that feature is absent. The builder therefore accumulates the
 * total into a dedicated record alongside the bins, and this simply reads it.
 */
inline void ExactNodeTotal(common::Span<double const> hist, bst_target_t n_free,
                           bst_bin_t n_total_bins, common::Span<double> out) {
  auto record_size = ExactHistRecordSize(n_free);
  CHECK_EQ(out.size(), record_size);
  CHECK_EQ(hist.size(), (static_cast<std::size_t>(n_total_bins) + 1) * record_size);
  auto const* total = hist.data() + static_cast<std::size_t>(n_total_bins) * record_size;
  std::copy(total, total + record_size, out.begin());
}

/**
 * @brief Scratch reused across nodes and features during enumeration.
 */
struct ExactEnumerateWorkspace {
  ExactStatBuffer left;
  ExactStatBuffer right;

  void Reset(bst_target_t n_free) {
    left.Reset(n_free);
    right.Reset(n_free);
  }
};

/**
 * @brief Enumerate splits of one feature under the exact objective.
 *
 * Mirrors the scalar enumerator's two-pass structure. The forward pass accumulates the left
 * child bin by bin and sends rows with a missing value to the right; the backward pass
 * accumulates the right child and sends them left. Running both is what lets a sparse
 * feature choose the better default direction.
 *
 * The complementary child is always derived as `parent - accumulated`, the same subtraction
 * the histogram uses, so left and right always reconcile to the parent exactly.
 */
inline void EnumerateExactFeature(common::ExactMultinomialLeafSolver* solver,
                                  TrainParam const& param, bst_target_t n_classes,
                                  common::HistogramCuts const& cut, bst_feature_t fidx,
                                  common::Span<double const> hist,
                                  common::PackedMultinomialStats<double const> parent,
                                  double parent_gain, bool may_have_missing,
                                  ExactEnumerateWorkspace* workspace,
                                  ExactSplitCandidate* p_best) {
  auto n_free = static_cast<bst_target_t>(n_classes - 1);
  auto record_size = ExactHistRecordSize(n_free);
  auto bin_begin = static_cast<bst_bin_t>(cut.Ptrs()[fidx]);
  auto bin_end = static_cast<bst_bin_t>(cut.Ptrs()[fidx + 1]);
  if (bin_end <= bin_begin) {
    return;
  }
  auto const& cut_values = cut.Values();

  auto accumulate = [&](bst_bin_t bin, common::Span<double> into) {
    auto const* record = hist.data() + static_cast<std::size_t>(bin) * record_size;
    for (std::size_t i = 0; i < record_size; ++i) {
      into[i] += record[i];
    }
  };
  auto complement = [&](common::Span<double const> accumulated, common::Span<double> into) {
    for (std::size_t i = 0; i < record_size; ++i) {
      into[i] = parent.Data()[i] - accumulated[i];
    }
  };

  // Forward: accumulate the left child, missing values go right.
  workspace->left.Zero();
  for (bst_bin_t bin = bin_begin; bin < bin_end - 1; ++bin) {
    accumulate(bin, workspace->left.Data());
    complement(common::Span<double const>{workspace->left.Data()}, workspace->right.Data());
    auto loss_chg =
        ExactSplitLossChange(solver, param, n_classes, parent_gain, workspace->left.ConstView(),
                             workspace->right.ConstView());
    if (loss_chg > p_best->loss_chg) {
      ExactSplitCandidate candidate;
      candidate.loss_chg = loss_chg;
      candidate.fidx = fidx;
      candidate.split_value = cut_values[bin];
      candidate.default_left = false;
      candidate.valid = true;
      p_best->Update(candidate);
    }
  }

  // Backward: accumulate the right child, missing values go left.
  //
  // On a dense matrix this pass is provably redundant: with no row taking the default
  // direction it enumerates exactly the same partitions as the forward pass, and
  // `default_left` cannot affect a single training row. It is the second of the two
  // factorizations per candidate, so skipping it roughly halves the enumeration cost.
  //
  // Whether rows are missing is read from the data (`GHistIndexMatrix::IsDense`), not
  // inferred by comparing floating point sums. An earlier version compared the node total
  // against the feature's bin sum with a relative tolerance; that made a split decision
  // depend on a magic epsilon, and a feature with a genuinely tiny missing mass would have
  // silently lost its default-left candidate. A structural test has no such failure mode.
  if (!may_have_missing) {
    return;
  }
  workspace->right.Zero();
  for (bst_bin_t bin = bin_end - 1; bin > bin_begin; --bin) {
    accumulate(bin, workspace->right.Data());
    complement(common::Span<double const>{workspace->right.Data()}, workspace->left.Data());
    auto loss_chg =
        ExactSplitLossChange(solver, param, n_classes, parent_gain, workspace->left.ConstView(),
                             workspace->right.ConstView());
    if (loss_chg > p_best->loss_chg) {
      ExactSplitCandidate candidate;
      candidate.loss_chg = loss_chg;
      candidate.fidx = fidx;
      // Splitting before `bin` puts `bin` and everything above it on the right. The
      // loop stops above bin_begin, so bin - 1 is always a valid cut of this feature.
      candidate.split_value = cut_values[bin - 1];
      candidate.default_left = true;
      candidate.valid = true;
      p_best->Update(candidate);
    }
  }
}

/**
 * @brief Rebuild the child statistics of an already chosen split.
 *
 * Lives beside @ref EnumerateExactFeature because it must mirror it exactly. The invariant
 * is that the statistics a leaf weight is solved from are the statistics whose gain selected
 * that split -- bit for bit, not to within a tolerance. Floating point addition is not
 * associative, so replaying the same bins in a different order is a different number.
 *
 * The enumerator accumulates the LEFT child over increasing bins on its forward pass and the
 * RIGHT child over decreasing bins on its backward pass, always deriving the sibling as
 * `parent - accumulated`. `default_left` records which pass won, so it selects the direction
 * to replay here.
 *
 * Selecting bins by comparing their cut value against `split_value` is equivalent to
 * selecting them by index because cut values are strictly increasing within a feature:
 * `WQuantileSketch::QueryCutValues` advances past equal values (`advance_to_next_distinct`
 * in `common/quantile.h`), so no two bins of one feature share a cut value.
 */
inline void ReconstructExactSplitChildren(common::HistogramCuts const& cut, bst_feature_t fidx,
                                          float split_value, bool default_left,
                                          common::Span<double const> hist,
                                          common::Span<double const> parent, bst_target_t n_free,
                                          ExactStatBuffer* left, ExactStatBuffer* right) {
  auto record_size = ExactHistRecordSize(n_free);
  CHECK_EQ(parent.size(), record_size);
  left->Zero();
  right->Zero();
  auto bin_begin = static_cast<bst_bin_t>(cut.Ptrs()[fidx]);
  auto bin_end = static_cast<bst_bin_t>(cut.Ptrs()[fidx + 1]);
  auto const& values = cut.Values();

  auto add = [&](bst_bin_t bin, common::Span<double> into) {
    auto const* record = hist.data() + static_cast<std::size_t>(bin) * record_size;
    for (std::size_t i = 0; i < record_size; ++i) {
      into[i] += record[i];
    }
  };

  common::Span<double> accumulated;
  common::Span<double> other;
  if (default_left) {
    accumulated = right->Data();
    other = left->Data();
    for (bst_bin_t bin = bin_end - 1; bin >= bin_begin && values[bin] > split_value; --bin) {
      add(bin, accumulated);
    }
  } else {
    accumulated = left->Data();
    other = right->Data();
    for (bst_bin_t bin = bin_begin; bin < bin_end && values[bin] <= split_value; ++bin) {
      add(bin, accumulated);
    }
  }
  for (std::size_t i = 0; i < record_size; ++i) {
    other[i] = parent[i] - accumulated[i];
  }
}

/**
 * @brief Best split for one node across the supplied features.
 *
 * Returns an invalid candidate when no feature yields a loss change above `min_split_loss`,
 * which is the caller's signal to make the node a leaf.
 */
[[nodiscard]] inline ExactSplitCandidate EnumerateExactNode(
    common::ExactMultinomialLeafSolver* solver, TrainParam const& param, bst_target_t n_classes,
    common::HistogramCuts const& cut, common::Span<double const> hist,
    common::Span<bst_feature_t const> features, bool may_have_missing,
    ExactEnumerateWorkspace* workspace, ExactStatBuffer* node_total) {
  auto n_free = static_cast<bst_target_t>(n_classes - 1);
  ExactNodeTotal(hist, n_free, cut.TotalBins(), node_total->Data());
  auto parent = node_total->ConstView();

  // The parent is constant for this node, so its gain -- an O(d^3) factorization -- is
  // computed once here rather than once per candidate bin.
  auto parent_gain = ExactParentGain(solver, param, n_classes, parent);

  ExactSplitCandidate best;
  for (auto fidx : features) {
    EnumerateExactFeature(solver, param, n_classes, cut, fidx, hist, parent, parent_gain,
                          may_have_missing, workspace, &best);
  }

  // gamma / min_split_loss lives in the same doubled units as the gain, so the comparison is
  // the scalar path's comparison unchanged.
  if (best.valid && best.loss_chg <= static_cast<double>(param.min_split_loss)) {
    best.valid = false;
  }
  return best;
}
}  // namespace xgboost::tree
#endif  // XGBOOST_TREE_HIST_EXACT_SPLIT_H_
