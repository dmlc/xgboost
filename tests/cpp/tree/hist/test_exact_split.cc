/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>
#include <xgboost/context.h>
#include <xgboost/gradient.h>

#include <algorithm>  // for max, nth_element
#include <cmath>      // for isnan, fabs
#include <limits>     // for numeric_limits
#include <cstddef>    // for size_t
#include <functional>  // for function
#include <memory>      // for shared_ptr
#include <numeric>    // for iota, accumulate
#include <utility>    // for move, make_pair
#include <vector>     // for vector

#include "../../../../src/data/gradient_index.h"
#include "../../../../src/tree/hist/exact_histogram.h"
#include "../../../../src/tree/hist/expand_entry.h"
#include "../../../../src/tree/driver.h"
#include "../../../../src/tree/hist/exact_split.h"
#include "../../helpers.h"

namespace xgboost::tree {
namespace {
TrainParam MakeParam(double lambda, double min_child_weight = 0.0, double gamma = 0.0) {
  TrainParam param;
  param.UpdateAllowUnknown(Args{{"reg_lambda", std::to_string(lambda)},
                                {"min_child_weight", std::to_string(min_child_weight)},
                                {"min_split_loss", std::to_string(gamma)}});
  return param;
}

/**
 * @brief A dataset whose single feature perfectly separates two class distributions.
 *
 * Rows below the threshold favour class 0, rows above favour class 1. A correct split
 * enumerator must find the boundary.
 */
struct SeparableProblem {
  std::shared_ptr<DMatrix> fmat;
  linalg::Matrix<GradientPair> gpair;
  ExactHessian hessian;
  std::size_t n_rows;
  bst_target_t n_classes;
};

SeparableProblem MakeSeparable(Context const* ctx, std::size_t n_rows, bst_target_t n_classes,
                               double signal, bool vary_labels = true) {
  SeparableProblem out;
  out.n_rows = n_rows;
  out.n_classes = n_classes;
  auto n_free = static_cast<bst_target_t>(n_classes - 1);

  // One informative feature holding a monotone ramp, so the bins are ordered and the class
  // boundary corresponds to a specific feature value.
  std::vector<float> data(n_rows);
  std::iota(data.begin(), data.end(), 0.0f);
  out.fmat = GetDMatrixFromData(data, n_rows, 1);

  out.gpair.Reshape(n_rows, n_classes);
  out.hessian.Reshape(n_rows, n_free);
  auto h_gpair = out.gpair.HostView();

  for (std::size_t r = 0; r < n_rows; ++r) {
    bool upper = r >= n_rows / 2;
    std::vector<double> p(n_classes, 1.0);
    // Shift mass towards class 0 below the boundary and class 1 above it.
    p[upper ? 1 : 0] += signal;
    double total = 0.0;
    for (auto v : p) {
      total += v;
    }
    for (auto& v : p) {
      v /= total;
    }
    std::size_t label = (vary_labels && upper) ? 1 : 0;

    for (bst_target_t t = 0; t < n_classes; ++t) {
      auto grad = static_cast<float>(p[t] - (label == t ? 1.0 : 0.0));
      h_gpair(r, t) = GradientPair{grad, std::max(std::fabs(grad), 1e-16f)};
    }
    auto row = out.hessian.HostRow(r);
    auto view = common::PackedHessianAtRow(row, n_free, 0);
    for (std::size_t i = 0; i < n_free; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        view.Set(i, j, static_cast<float>(p[i] * ((i == j ? 1.0 : 0.0) - p[j])));
      }
    }
  }
  return out;
}

/** @brief Build the exact root histogram for a problem. */
void BuildRoot(Context const* ctx, SeparableProblem const& problem, GHistIndexMatrix const& gmat,
               ExactHistCollection* hist) {
  auto n_free = static_cast<bst_target_t>(problem.n_classes - 1);
  hist->Reset(gmat.cut.TotalBins(), n_free, 16);
  hist->AllocateHistograms(std::vector<bst_node_t>{0});
  ZeroExactHist((*hist)[0]);

  std::vector<bst_idx_t> rows(problem.n_rows);
  std::iota(rows.begin(), rows.end(), 0);
  ExactHistThreadBuffer buffer;
  BuildExactHist(ctx, (*hist)[0], n_free, gmat, common::Span<bst_idx_t const>{rows},
                 problem.gpair.HostView(), problem.hessian, &buffer);
}
}  // anonymous namespace

/**
 * The node total is what the enumerator subtracts from, so it must equal the sum of the rows.
 *
 * Deliberately *not* named after the bin sum: the two coincide here only because there is a
 * single feature. `ExactHistogram.NodeTotalCountsEachRowOnce` covers the general case, where
 * the bin sum over-counts and the node total does not.
 */
TEST(ExactSplit, NodeTotalMatchesPerRowSum) {
  Context ctx;
  bst_target_t constexpr kNumClasses = 3;
  auto problem = MakeSeparable(&ctx, 64, kNumClasses, 2.0);
  auto const& gmat =
      *(problem.fmat->GetBatches<GHistIndexMatrix>(&ctx, BatchParam{8, 0.5}).begin());

  ExactHistCollection hist;
  BuildRoot(&ctx, problem, gmat, &hist);

  auto n_free = static_cast<bst_target_t>(kNumClasses - 1);
  ExactStatBuffer total;
  total.Reset(n_free);
  ExactNodeTotal(common::Span<double const>{hist[0]}, n_free, gmat.cut.TotalBins(), total.Data());

  // Compare against the sum of the per-row statistics directly.
  auto record_size = ExactHistRecordSize(n_free);
  std::vector<double> reference(record_size, 0.0);
  auto h_gpair = problem.gpair.HostView();
  for (std::size_t r = 0; r < problem.n_rows; ++r) {
    for (bst_target_t i = 0; i < n_free; ++i) {
      reference[i] += h_gpair(r, i).GetGrad();
    }
    auto row = problem.hessian.HostRow(r);
    for (std::size_t k = 0; k < row.size(); ++k) {
      reference[n_free + k] += row[k];
    }
  }
  // One feature, so every row contributes to exactly one bin.
  for (std::size_t i = 0; i < record_size; ++i) {
    EXPECT_NEAR(total.Data()[i], reference[i], 1e-6) << "entry " << i;
  }
}

/** A separable feature must produce a split at the true boundary. */
TEST(ExactSplit, FindsSeparatingBoundary) {
  Context ctx;
  bst_target_t constexpr kNumClasses = 3;
  std::size_t constexpr kRows = 128;
  auto problem = MakeSeparable(&ctx, kRows, kNumClasses, 4.0);
  auto const& gmat =
      *(problem.fmat->GetBatches<GHistIndexMatrix>(&ctx, BatchParam{16, 0.5}).begin());

  ExactHistCollection hist;
  BuildRoot(&ctx, problem, gmat, &hist);

  auto n_free = static_cast<bst_target_t>(kNumClasses - 1);
  common::ExactMultinomialLeafSolver solver{n_free};
  ExactEnumerateWorkspace workspace;
  workspace.Reset(n_free);
  ExactStatBuffer total;
  total.Reset(n_free);

  auto param = MakeParam(1.0);
  std::vector<bst_feature_t> features{0};
  auto best = EnumerateExactNode(&solver, param, kNumClasses, gmat.cut,
                                 common::Span<double const>{hist[0]},
                                 common::Span<bst_feature_t const>{features},
                                 /*may_have_missing=*/true, &workspace, &total);

  ASSERT_TRUE(best.valid);
  EXPECT_EQ(best.fidx, 0);
  EXPECT_GT(best.loss_chg, 0.0);
  // The boundary sits at row kRows/2, i.e. feature value kRows/2.
  EXPECT_NEAR(best.split_value, static_cast<float>(kRows / 2), static_cast<float>(kRows) / 8.0f)
      << "split landed far from the true class boundary";
}

/** gamma gates expansion in the same units as the gain. */
TEST(ExactSplit, GammaGatesExpansion) {
  Context ctx;
  bst_target_t constexpr kNumClasses = 3;
  auto problem = MakeSeparable(&ctx, 128, kNumClasses, 4.0);
  auto const& gmat =
      *(problem.fmat->GetBatches<GHistIndexMatrix>(&ctx, BatchParam{16, 0.5}).begin());

  ExactHistCollection hist;
  BuildRoot(&ctx, problem, gmat, &hist);
  auto n_free = static_cast<bst_target_t>(kNumClasses - 1);
  common::ExactMultinomialLeafSolver solver{n_free};
  ExactEnumerateWorkspace workspace;
  workspace.Reset(n_free);
  ExactStatBuffer total;
  total.Reset(n_free);
  std::vector<bst_feature_t> features{0};

  auto enumerate = [&](TrainParam const& param) {
    return EnumerateExactNode(&solver, param, kNumClasses, gmat.cut,
                              common::Span<double const>{hist[0]},
                              common::Span<bst_feature_t const>{features},
                              /*may_have_missing=*/true, &workspace, &total);
  };

  auto permissive = enumerate(MakeParam(1.0, 0.0, 0.0));
  ASSERT_TRUE(permissive.valid);
  auto achievable = permissive.loss_chg;

  // gamma just below the achievable gain still allows the split.
  auto allowed = enumerate(MakeParam(1.0, 0.0, achievable * 0.5));
  EXPECT_TRUE(allowed.valid);

  // gamma above it suppresses the split entirely.
  auto blocked = enumerate(MakeParam(1.0, 0.0, achievable * 2.0));
  EXPECT_FALSE(blocked.valid);
}

/** min_child_weight gates expansion through the reference-invariant curvature. */
TEST(ExactSplit, MinChildWeightGatesExpansion) {
  Context ctx;
  bst_target_t constexpr kNumClasses = 3;
  auto problem = MakeSeparable(&ctx, 128, kNumClasses, 4.0);
  auto const& gmat =
      *(problem.fmat->GetBatches<GHistIndexMatrix>(&ctx, BatchParam{16, 0.5}).begin());

  ExactHistCollection hist;
  BuildRoot(&ctx, problem, gmat, &hist);
  auto n_free = static_cast<bst_target_t>(kNumClasses - 1);
  common::ExactMultinomialLeafSolver solver{n_free};
  ExactEnumerateWorkspace workspace;
  workspace.Reset(n_free);
  ExactStatBuffer total;
  total.Reset(n_free);
  std::vector<bst_feature_t> features{0};

  auto enumerate = [&](double min_child_weight) {
    return EnumerateExactNode(&solver, MakeParam(1.0, min_child_weight), kNumClasses, gmat.cut,
                              common::Span<double const>{hist[0]},
                              common::Span<bst_feature_t const>{features},
                              /*may_have_missing=*/true, &workspace, &total);
  };

  EXPECT_TRUE(enumerate(0.0).valid);
  // The whole node's per-class curvature bounds any child's, so a threshold above it must
  // reject every candidate.
  auto node_curvature = ExactChildCurvature(total.ConstView(), kNumClasses);
  EXPECT_GT(node_curvature, 0.0);
  EXPECT_FALSE(enumerate(node_curvature * 1.01).valid);
}

/**
 * A feature carrying no information yields no useful split.
 *
 * "No information" means both the predicted distribution and the label are identical on
 * either side of every candidate boundary. Note that uniform *probabilities* alone are not
 * enough: if the labels still differ, the gradient carries a strong signal and the split is
 * highly informative, which is the correct behaviour.
 */
TEST(ExactSplit, NoSplitWhenFeatureCarriesNoSignal) {
  Context ctx;
  bst_target_t constexpr kNumClasses = 3;
  // Identical p and identical label for every row: nothing distinguishes the halves.
  auto flat = MakeSeparable(&ctx, 128, kNumClasses, 0.0, /*vary_labels=*/false);
  auto const& gmat =
      *(flat.fmat->GetBatches<GHistIndexMatrix>(&ctx, BatchParam{16, 0.5}).begin());

  ExactHistCollection hist;
  BuildRoot(&ctx, flat, gmat, &hist);
  auto n_free = static_cast<bst_target_t>(kNumClasses - 1);
  common::ExactMultinomialLeafSolver solver{n_free};
  ExactEnumerateWorkspace workspace;
  workspace.Reset(n_free);
  ExactStatBuffer total;
  total.Reset(n_free);
  std::vector<bst_feature_t> features{0};

  // lambda = 0 makes the gain homogeneous of degree one in the statistics, so splitting a
  // homogeneous node into proportional halves must recover exactly the parent's gain.
  auto best = EnumerateExactNode(&solver, MakeParam(0.0), kNumClasses, gmat.cut,
                                 common::Span<double const>{hist[0]},
                                 common::Span<bst_feature_t const>{features},
                                 /*may_have_missing=*/true, &workspace, &total);
  if (best.valid) {
    EXPECT_NEAR(best.loss_chg, 0.0, 1e-6)
        << "a homogeneous node reported a real gain from splitting";
  }
}

/**
 * A better-fitted region has less left to gain, which is a property of the objective rather
 * than of the enumerator. Recording it guards against sign or convention drift.
 */
TEST(ExactSplit, WellFittedRegionGainsLess) {
  Context ctx;
  bst_target_t constexpr kNumClasses = 3;
  auto const& gmat_src = MakeSeparable(&ctx, 128, kNumClasses, 4.0);
  auto const& gmat =
      *(gmat_src.fmat->GetBatches<GHistIndexMatrix>(&ctx, BatchParam{16, 0.5}).begin());
  auto n_free = static_cast<bst_target_t>(kNumClasses - 1);
  common::ExactMultinomialLeafSolver solver{n_free};
  ExactEnumerateWorkspace workspace;
  workspace.Reset(n_free);
  std::vector<bst_feature_t> features{0};

  auto gain_for = [&](double signal) {
    auto problem = MakeSeparable(&ctx, 128, kNumClasses, signal);
    ExactHistCollection hist;
    BuildRoot(&ctx, problem, gmat, &hist);
    ExactStatBuffer total;
    total.Reset(n_free);
    auto best = EnumerateExactNode(&solver, MakeParam(1.0), kNumClasses, gmat.cut,
                                   common::Span<double const>{hist[0]},
                                   common::Span<bst_feature_t const>{features},
                                   /*may_have_missing=*/true, &workspace, &total);
    EXPECT_TRUE(best.valid);
    return best.loss_chg;
  };

  // signal = 0: predictions are uniform while labels are split, so the residual is large.
  // signal = 4: predictions already track the labels, so little remains to be gained.
  EXPECT_GT(gain_for(0.0), gain_for(4.0));
}

/** Off-diagonal Hessian terms must actually influence the chosen split. */
TEST(ExactSplit, OffDiagonalTermsAffectTheDecision) {
  Context ctx;
  bst_target_t constexpr kNumClasses = 3;
  std::size_t constexpr kRows = 128;
  auto problem = MakeSeparable(&ctx, kRows, kNumClasses, 4.0);
  auto const& gmat =
      *(problem.fmat->GetBatches<GHistIndexMatrix>(&ctx, BatchParam{16, 0.5}).begin());

  ExactHistCollection hist;
  BuildRoot(&ctx, problem, gmat, &hist);
  auto n_free = static_cast<bst_target_t>(kNumClasses - 1);
  common::ExactMultinomialLeafSolver solver{n_free};
  ExactEnumerateWorkspace workspace;
  workspace.Reset(n_free);
  ExactStatBuffer total;
  total.Reset(n_free);
  std::vector<bst_feature_t> features{0};
  auto param = MakeParam(1.0);

  auto with_coupling =
      EnumerateExactNode(&solver, param, kNumClasses, gmat.cut, common::Span<double const>{hist[0]},
                         common::Span<bst_feature_t const>{features},
                         /*may_have_missing=*/true, &workspace, &total);
  ASSERT_TRUE(with_coupling.valid);

  // Strip the off-diagonal coupling from every bin, leaving a purely diagonal Hessian, and
  // re-enumerate. The gain must change: if it did not, the dense Hessian would not be
  // affecting the decision at all.
  for (bst_bin_t b = 0; b < gmat.cut.TotalBins(); ++b) {
    auto record = hist.RecordAt(0, b);
    for (std::size_t i = 0; i < n_free; ++i) {
      for (std::size_t j = 0; j < i; ++j) {
        record.SetHessian(i, j, 0.0);
      }
    }
  }
  ExactStatBuffer diag_total;
  diag_total.Reset(n_free);
  auto diagonal_only =
      EnumerateExactNode(&solver, param, kNumClasses, gmat.cut, common::Span<double const>{hist[0]},
                         common::Span<bst_feature_t const>{features},
                         /*may_have_missing=*/true, &workspace, &diag_total);
  ASSERT_TRUE(diagonal_only.valid);
  EXPECT_NE(with_coupling.loss_chg, diagonal_only.loss_chg)
      << "removing the off-diagonal Hessian left the gain unchanged, so the dense terms are "
         "not reaching the split decision";
}

namespace {
/**
 * @brief Build gpair and the exact sidecar for an arbitrary matrix.
 *
 * The caller supplies the per-row class distribution and label, which lets a test couple the
 * signal to something structural -- such as whether a feature is present.
 */
template <typename MakeRow>
SeparableProblem MakeProblemFor(std::shared_ptr<DMatrix> fmat, std::size_t n_rows,
                                bst_target_t n_classes, MakeRow make_row,
                                std::function<double(std::size_t)> weight = {}) {
  SeparableProblem out;
  out.fmat = std::move(fmat);
  out.n_rows = n_rows;
  out.n_classes = n_classes;
  auto n_free = static_cast<bst_target_t>(n_classes - 1);

  out.gpair.Reshape(n_rows, n_classes);
  out.hessian.Reshape(n_rows, n_free);
  auto h_gpair = out.gpair.HostView();

  for (std::size_t r = 0; r < n_rows; ++r) {
    auto row = make_row(r);
    auto const& p = row.first;
    auto label = row.second;
    auto w = weight ? weight(r) : 1.0;
    for (bst_target_t t = 0; t < n_classes; ++t) {
      auto grad = static_cast<float>(w * (p[t] - (label == t ? 1.0 : 0.0)));
      h_gpair(r, t) = GradientPair{grad, std::max(std::fabs(grad), 1e-16f)};
    }
    auto view = common::PackedHessianAtRow(out.hessian.HostRow(r), n_free, 0);
    for (std::size_t i = 0; i < n_free; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        view.Set(i, j, static_cast<float>(w * p[i] * ((i == j ? 1.0 : 0.0) - p[j])));
      }
    }
  }
  return out;
}

/** @brief A distribution concentrated on `favoured`, with the rest spread evenly. */
std::vector<double> Peaked(bst_target_t n_classes, std::size_t favoured, double signal) {
  std::vector<double> p(n_classes, 1.0);
  p[favoured] += signal;
  auto total = std::accumulate(p.cbegin(), p.cend(), 0.0);
  for (auto& v : p) {
    v /= total;
  }
  return p;
}

/**
 * @brief Value of feature `fidx` per row, or NaN where the row does not carry it.
 *
 * Read from the source matrix rather than assumed, because the generator decides both which
 * entries are present and what they hold.
 */
std::vector<float> FeatureColumn(DMatrix* fmat, bst_feature_t fidx, std::size_t n_rows) {
  std::vector<float> column(n_rows, std::numeric_limits<float>::quiet_NaN());
  for (auto const& page : fmat->GetBatches<SparsePage>()) {
    auto batch = page.GetView();
    for (std::size_t r = 0; r < batch.Size(); ++r) {
      for (auto const& e : batch[r]) {
        if (e.index == fidx) {
          column[page.base_rowid + r] = e.fvalue;
        }
      }
    }
  }
  return column;
}

struct Enumerated {
  ExactSplitCandidate with_backward;
  ExactSplitCandidate forward_only;
};

Enumerated EnumerateBothWays(Context const* ctx, SeparableProblem const& problem,
                             GHistIndexMatrix const& gmat, TrainParam const& param,
                             std::vector<bst_feature_t> const& features) {
  auto n_free = static_cast<bst_target_t>(problem.n_classes - 1);
  ExactHistCollection hist;
  BuildRoot(ctx, problem, gmat, &hist);

  common::ExactMultinomialLeafSolver solver{n_free};
  ExactEnumerateWorkspace workspace;
  workspace.Reset(n_free);
  ExactStatBuffer total;
  total.Reset(n_free);

  auto run = [&](bool may_have_missing) {
    return EnumerateExactNode(&solver, param, problem.n_classes, gmat.cut,
                              common::Span<double const>{hist[0]},
                              common::Span<bst_feature_t const>{features}, may_have_missing,
                              &workspace, &total);
  };
  return Enumerated{run(true), run(false)};
}
}  // anonymous namespace

/**
 * Phase 9. Skipping the backward pass on a dense matrix is a claim about the data, not a
 * tolerance: with no missing values the backward pass re-enumerates the same partitions, so
 * the shortcut must change nothing at all. Equality here is exact, not approximate.
 */
TEST(ExactSplit, DenseMatrixMakesTheBackwardPassRedundant) {
  Context ctx;
  bst_target_t constexpr kNumClasses = 4;
  std::size_t constexpr kRows = 256;
  bst_feature_t constexpr kCols = 5;

  auto fmat = RandomDataGenerator{kRows, kCols, 0.0}.Seed(31).GenerateDMatrix();
  auto problem = MakeProblemFor(fmat, kRows, kNumClasses, [&](std::size_t r) {
    return std::make_pair(Peaked(kNumClasses, r % kNumClasses, 3.0 + 0.01 * (r % 7)),
                          static_cast<std::size_t>((r * 3) % kNumClasses));
  });
  auto const& gmat =
      *(problem.fmat->GetBatches<GHistIndexMatrix>(&ctx, BatchParam{16, 0.5}).begin());
  // The predicate the production shortcut reads. If a sparsity-0 matrix ever stopped
  // reporting dense, the shortcut would be taken on data that needs both passes.
  ASSERT_TRUE(gmat.IsDense());
  // The builder reads the DMatrix-level predicate, because a dataset can span several pages
  // and only some of them need be dense. For a single page the two must agree.
  ASSERT_EQ(problem.fmat->IsDense(), gmat.IsDense());

  std::vector<bst_feature_t> features(kCols);
  std::iota(features.begin(), features.end(), 0);
  auto result = EnumerateBothWays(&ctx, problem, gmat, MakeParam(1.0), features);

  ASSERT_TRUE(result.with_backward.valid);
  ASSERT_TRUE(result.forward_only.valid);
  EXPECT_EQ(result.forward_only.fidx, result.with_backward.fidx);
  EXPECT_EQ(result.forward_only.split_value, result.with_backward.split_value);
  EXPECT_EQ(result.forward_only.default_left, result.with_backward.default_left);
  // Bit-for-bit: the winning candidate is produced by the same accumulation either way.
  EXPECT_EQ(result.forward_only.loss_chg, result.with_backward.loss_chg);
}

/**
 * Phase 9, the other half. The flag must be genuinely load bearing on sparse data: if
 * skipping the backward pass could never change an answer, the shortcut would be vacuous and
 * this test would be the one to catch that.
 */
TEST(ExactSplit, SparseMatrixNeedsTheBackwardPass) {
  Context ctx;
  bst_target_t constexpr kNumClasses = 3;
  std::size_t constexpr kRows = 512;
  bst_feature_t constexpr kCols = 2;

  auto fmat = RandomDataGenerator{kRows, kCols, 0.5}.Seed(17).GenerateDMatrix();
  auto column = FeatureColumn(fmat.get(), 0, kRows);
  std::vector<float> observed;
  for (auto v : column) {
    if (!std::isnan(v)) {
      observed.push_back(v);
    }
  }
  ASSERT_FALSE(observed.empty());
  ASSERT_LT(observed.size(), kRows);
  // The median of the values actually present, so the class boundary lands inside the
  // feature's range rather than at an arbitrary row index. Bins are ordered by value, not
  // by row, which is what makes this the boundary the enumerator can actually cut at.
  std::nth_element(observed.begin(), observed.begin() + observed.size() / 2, observed.end());
  auto median = observed[observed.size() / 2];

  // Rows missing feature 0 belong with its *low* end, so the best partition requires sending
  // missing values left -- a candidate only the backward pass can produce.
  auto problem = MakeProblemFor(fmat, kRows, kNumClasses, [&](std::size_t r) {
    std::size_t favoured = (!std::isnan(column[r]) && column[r] >= median) ? 1u : 0u;
    return std::make_pair(Peaked(kNumClasses, favoured, 6.0), favoured);
  });

  auto const& gmat =
      *(problem.fmat->GetBatches<GHistIndexMatrix>(&ctx, BatchParam{32, 0.5}).begin());
  ASSERT_FALSE(gmat.IsDense());
  ASSERT_EQ(problem.fmat->IsDense(), gmat.IsDense());

  std::vector<bst_feature_t> features{0};
  auto result = EnumerateBothWays(&ctx, problem, gmat, MakeParam(1.0), features);

  ASSERT_TRUE(result.with_backward.valid);
  ASSERT_TRUE(result.forward_only.valid);
  // The backward pass only ever adds candidates, so it can never lose to forward-only.
  EXPECT_GT(result.with_backward.loss_chg, result.forward_only.loss_chg)
      << "the backward pass changed nothing on a sparse feature engineered to need it; "
         "either the default direction is not reaching the decision, or the construction "
         "no longer separates missing rows";
  EXPECT_TRUE(result.with_backward.default_left)
      << "the winning split should send missing rows to the low side";
}

/**
 * Phase 10. On a dense matrix the two passes evaluate identical partitions, so every
 * candidate ties. `Update` requires a strict improvement, so the forward candidate wins and
 * the default direction is decided by the enumeration order, not by rounding noise.
 */
TEST(ExactSplit, TiedDefaultDirectionResolvesDeterministically) {
  Context ctx;
  bst_target_t constexpr kNumClasses = 3;
  std::size_t constexpr kRows = 128;

  auto problem = MakeSeparable(&ctx, kRows, kNumClasses, 4.0);
  auto const& gmat =
      *(problem.fmat->GetBatches<GHistIndexMatrix>(&ctx, BatchParam{16, 0.5}).begin());
  ASSERT_TRUE(gmat.IsDense());

  std::vector<bst_feature_t> features{0};
  ExactSplitCandidate first;
  for (int repeat = 0; repeat < 3; ++repeat) {
    auto result = EnumerateBothWays(&ctx, problem, gmat, MakeParam(1.0), features);
    ASSERT_TRUE(result.with_backward.valid);
    // Running the full two-pass enumeration over tied candidates keeps the forward one.
    EXPECT_FALSE(result.with_backward.default_left);
    if (repeat == 0) {
      first = result.with_backward;
    } else {
      EXPECT_EQ(result.with_backward.fidx, first.fidx);
      EXPECT_EQ(result.with_backward.split_value, first.split_value);
      EXPECT_EQ(result.with_backward.default_left, first.default_left);
      EXPECT_EQ(result.with_backward.loss_chg, first.loss_chg);
    }
  }
}

/**
 * The invariant that ties enumeration to leaf weights: the statistics a leaf is solved from
 * must be the statistics whose gain selected the split.
 *
 * Two things are checked, and the test guards its own sensitivity so it cannot go vacuous:
 *
 *  1. A SENSITIVITY GUARD computes the right child both upward and downward and asserts the
 *     two orders actually produce different doubles on this fixture. Reordering a sum only
 *     changes the result when the roundings differ; on a fixture where they happen to agree,
 *     any ordering bug would be undetectable and this test would prove nothing. If the guard
 *     ever stops holding, the test fails loudly instead of passing silently.
 *  2. The production reconstruction is then required to equal the DOWNWARD order exactly --
 *     the order `EnumerateExactFeature`'s backward pass uses -- and to differ from the
 *     upward one.
 *
 * Finally the reconstructed children are fed back through `ExactSplitLossChange` and must
 * reproduce the winning `loss_chg` exactly, which is the semantic form of the invariant and
 * does not re-implement any accumulation.
 */
TEST(ExactSplit, ReconstructedChildrenReproduceTheWinningGainExactly) {
  Context ctx;
  bst_target_t constexpr kNumClasses = 3;
  std::size_t constexpr kRows = 2048;
  auto n_free = static_cast<bst_target_t>(kNumClasses - 1);
  auto record_size = ExactHistRecordSize(n_free);

  struct Case {
    char const* what;
    float sparsity;
    bool expect_default_left;
  };
  for (auto const& c : {Case{"dense / forward pass", 0.0f, false},
                        Case{"sparse / backward pass", 0.5f, true}}) {
    auto fmat = RandomDataGenerator{kRows, 2, c.sparsity}.Seed(17).GenerateDMatrix();
    auto column = FeatureColumn(fmat.get(), 0, kRows);
    std::vector<float> observed;
    for (auto v : column) {
      if (!std::isnan(v)) {
        observed.push_back(v);
      }
    }
    ASSERT_FALSE(observed.empty());
    std::nth_element(observed.begin(), observed.begin() + observed.size() / 2, observed.end());
    auto median = observed[observed.size() / 2];
    auto problem = MakeProblemFor(
        fmat, kRows, kNumClasses,
        [&](std::size_t r) {
          std::size_t favoured = (!std::isnan(column[r]) && column[r] >= median) ? 1u : 0u;
          return std::make_pair(Peaked(kNumClasses, favoured, 6.0), favoured);
        },
        // Weights spanning eight orders of magnitude, so the bins carry a broad range and
        // the summation order is genuinely observable instead of cancelling out. Sample
        // weights like this are ordinary user input, not a contrivance.
        [](std::size_t r) { return std::pow(10.0, static_cast<double>(r % 9) - 4.0); });

    auto const& gmat =
        *(problem.fmat->GetBatches<GHistIndexMatrix>(&ctx, BatchParam{64, 0.5}).begin());
    ExactHistCollection hist;
    BuildRoot(&ctx, problem, gmat, &hist);

    common::ExactMultinomialLeafSolver solver{n_free};
    ExactEnumerateWorkspace workspace;
    workspace.Reset(n_free);
    ExactStatBuffer total;
    total.Reset(n_free);
    std::vector<bst_feature_t> features{0};
    auto param = MakeParam(1.0);
    auto best = EnumerateExactNode(&solver, param, kNumClasses, gmat.cut,
                                   common::Span<double const>{hist[0]},
                                   common::Span<bst_feature_t const>{features},
                                   /*may_have_missing=*/true, &workspace, &total);
    ASSERT_TRUE(best.valid) << c.what;
    ASSERT_EQ(best.default_left, c.expect_default_left)
        << c.what << ": the fixture no longer exercises the intended enumeration pass";

    // ---- sensitivity guard -------------------------------------------------
    auto bin_begin = static_cast<bst_bin_t>(gmat.cut.Ptrs()[best.fidx]);
    auto bin_end = static_cast<bst_bin_t>(gmat.cut.Ptrs()[best.fidx + 1]);
    auto const& values = gmat.cut.Values();
    auto selected = [&](bst_bin_t bin) {
      return best.default_left ? values[bin] > best.split_value
                               : values[bin] <= best.split_value;
    };
    std::vector<double> downward(record_size, 0.0);
    std::vector<double> upward(record_size, 0.0);
    int n_selected = 0;
    for (bst_bin_t bin = bin_end - 1; bin >= bin_begin; --bin) {
      if (!selected(bin)) {
        continue;
      }
      ++n_selected;
      for (std::size_t i = 0; i < record_size; ++i) {
        downward[i] += hist[0][static_cast<std::size_t>(bin) * record_size + i];
      }
    }
    for (bst_bin_t bin = bin_begin; bin < bin_end; ++bin) {
      if (!selected(bin)) {
        continue;
      }
      for (std::size_t i = 0; i < record_size; ++i) {
        upward[i] += hist[0][static_cast<std::size_t>(bin) * record_size + i];
      }
    }
    ASSERT_GT(n_selected, 2) << c.what << ": too few bins for the order to matter";
    bool orders_differ = false;
    for (std::size_t i = 0; i < record_size; ++i) {
      if (downward[i] != upward[i]) {
        orders_differ = true;
        break;
      }
    }
    ASSERT_TRUE(orders_differ)
        << c.what << ": the two summation orders agree bit-for-bit on this fixture ("
        << n_selected << " bins), so this test could not detect an ordering regression";

    // ---- the production reconstruction ------------------------------------
    ExactStatBuffer left;
    ExactStatBuffer right;
    left.Reset(n_free);
    right.Reset(n_free);
    ReconstructExactSplitChildren(gmat.cut, best.fidx, best.split_value, best.default_left,
                                  common::Span<double const>{hist[0]},
                                  common::Span<double const>{total.Data()}, n_free, &left,
                                  &right);
    // The enumerator accumulates the LEFT child upward on its forward pass and the RIGHT
    // child downward on its backward pass; the reconstruction must match whichever ran.
    auto accumulated = best.default_left ? right.Data() : left.Data();
    auto const& expected = best.default_left ? downward : upward;
    auto const& wrong_order = best.default_left ? upward : downward;
    for (std::size_t i = 0; i < record_size; ++i) {
      EXPECT_EQ(accumulated[i], expected[i])
          << c.what << " entry " << i << ": wrong accumulation order";
    }
    bool differs_from_wrong = false;
    for (std::size_t i = 0; i < record_size; ++i) {
      if (accumulated[i] != wrong_order[i]) {
        differs_from_wrong = true;
        break;
      }
    }
    EXPECT_TRUE(differs_from_wrong)
        << c.what << ": reconstruction matches BOTH orders, so the check is vacuous";

    // ---- the semantic invariant -------------------------------------------
    auto parent_gain = ExactParentGain(&solver, param, kNumClasses, total.ConstView());
    auto recomputed = ExactSplitLossChange(&solver, param, kNumClasses, parent_gain,
                                           left.ConstView(), right.ConstView());
    EXPECT_EQ(recomputed, best.loss_chg)
        << c.what << ": the reconstructed children do not reproduce the gain that selected "
                     "this split, so the leaf weight comes from different statistics";
  }
}

/**
 * The inherited `kRtEps` expansion gate, analysed for the exact path.
 *
 * `Driver::Push` and `IsValidExpandEntry` reject a candidate whose `loss_chg <= kRtEps`
 * (1e-6f, absolute). That threshold was written for the scalar path, so the question is
 * whether it still means the same thing when the gain is `G^T (H+R)^-1 G`.
 *
 * It does, for one reason: both paths express gain in the SAME units. XGBoost's
 * `CalcGainGivenWeight` returns `-2 dL`, and `ExactSplitLossChange` returns
 * `gain(left) + gain(right) - gain(parent)` in that same `-2 dL` convention. The gate is
 * therefore a "reject numerically-zero improvements" guard in loss units, not a
 * curvature-dependent quantity, and no rescaling is warranted.
 *
 * Two things still need checking rather than asserting, and both are checked here:
 *  1. `loss_chg` is narrowed to `float` before the comparison
 *     (`exact_builder.h`: `entry.split.loss_chg = static_cast<float>(best.loss_chg)`), so
 *     the narrowing must not move a value across the threshold.
 *  2. Acceptance must be monotone through the boundary, with no gap or double-count.
 */
TEST(ExactSplit, InheritedKRtEpsGateBehavesConsistentlyForExactGains) {
  TrainParam param;
  param.UpdateAllowUnknown(Args{{"max_depth", "6"}, {"min_split_loss", "0"}});

  // Values straddling kRtEps, as doubles, the way ExactSplitLossChange produces them.
  std::vector<double> const probes{0.5e-6, 0.9e-6,  0.99e-6, 1.0e-6,
                                   1.01e-6, 1.1e-6, 2.0e-6};
  bool previous_accepted = false;
  bool seen_accept = false;
  for (auto gain : probes) {
    // Match the float storage used by the split entry before applying the inherited kRtEps gate.
    auto narrowed = static_cast<float>(gain);

    MultiExpandEntry entry{0, 0};
    entry.split.loss_chg = narrowed;
    auto accepted = IsValidExpandEntry(entry, param, /*num_leaves=*/1);

    // The documented contract: strictly greater than kRtEps is accepted.
    EXPECT_EQ(accepted, narrowed > kRtEps)
        << "gain " << gain << " (float " << narrowed << ") was not gated as documented";

    // Monotone: once accepted, never rejected again as the gain grows.
    if (previous_accepted) {
      EXPECT_TRUE(accepted) << "acceptance is not monotone in the gain at " << gain;
    }
    previous_accepted = accepted;
    seen_accept |= accepted;
  }
  EXPECT_TRUE(seen_accept) << "no probe was accepted; the sweep does not cross the threshold";
  EXPECT_FALSE(static_cast<float>(0.5e-6) > kRtEps) << "the sweep does not start below it";

  // And gamma composes with the gate as it always did: a gain above kRtEps but below
  // min_split_loss is still rejected, by gamma rather than by the epsilon.
  TrainParam gated;
  gated.UpdateAllowUnknown(Args{{"max_depth", "6"}, {"min_split_loss", "0.5"}});
  MultiExpandEntry entry{0, 0};
  entry.split.loss_chg = 1e-3f;
  EXPECT_FALSE(IsValidExpandEntry(entry, gated, 1));
  EXPECT_TRUE(IsValidExpandEntry(entry, param, 1));
}
}  // namespace xgboost::tree
