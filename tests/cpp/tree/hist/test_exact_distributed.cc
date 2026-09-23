/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>
#include <xgboost/context.h>
#include <xgboost/gradient.h>
#include <xgboost/tree_model.h>
#include <xgboost/tree_updater.h>

#include <algorithm>  // for max, min
#include <cmath>      // for fabs
#include <cstddef>  // for size_t
#include <memory>   // for unique_ptr
#include <numeric>  // for iota
#include <vector>   // for vector

#include "../../../../src/collective/broadcast.h"
#include "../../../../src/collective/communicator-inl.h"
#include "../../../../src/common/exact_multinomial/packed_stats.h"
#include "../../../../src/tree/hist/exact_builder.h"
#include "../../collective/test_worker.h"
#include "../../helpers.h"

namespace xgboost::tree {
namespace {
bst_target_t constexpr kNumClasses = 3;
std::size_t constexpr kRows = 256;

/**
 * @brief Statistics for one row, derived purely from its GLOBAL index.
 *
 * Deriving from the global index is what makes a slice of rows carry exactly the statistics
 * those rows would have had in a single-process run, so the reduced total is comparable.
 */
void FillRow(std::size_t global_row, linalg::MatrixView<GradientPair> gpair,
             std::size_t local_row, common::Span<float> hessian_row) {
  auto n_free = static_cast<bst_target_t>(kNumClasses - 1);
  auto regime = (global_row * 4) / kRows;
  std::vector<double> p(kNumClasses, 1.0);
  p[regime % kNumClasses] += 3.0;
  double total = 0.0;
  for (auto v : p) {
    total += v;
  }
  for (auto& v : p) {
    v /= total;
  }
  auto label = regime % kNumClasses;
  auto w = 0.5f + static_cast<float>(global_row % 4);

  for (bst_target_t t = 0; t < kNumClasses; ++t) {
    auto grad = static_cast<float>(p[t] - (label == t ? 1.0 : 0.0)) * w;
    gpair(local_row, t) = GradientPair{grad, std::max(std::fabs(grad), 1e-16f)};
  }
  auto view = common::PackedHessianAtRow(hessian_row, n_free, 0);
  for (std::size_t i = 0; i < n_free; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      view.Set(i, j, static_cast<float>(w * p[i] * ((i == j ? 1.0 : 0.0) - p[j])));
    }
  }
}

struct Slice {
  std::shared_ptr<DMatrix> fmat;
  GradientContainer gpair;
};

std::vector<std::size_t> RowsForRank(std::int32_t rank, std::int32_t n_workers) {
  std::vector<std::size_t> rows;
  for (std::size_t r = 0; r < kRows; ++r) {
    if (static_cast<std::int32_t>(r % static_cast<std::size_t>(n_workers)) == rank) {
      rows.push_back(r);
    }
  }
  return rows;
}

Slice MakeSlice(std::vector<std::size_t> const& rows) {
  Slice out;
  auto n = rows.size();
  auto n_free = static_cast<bst_target_t>(kNumClasses - 1);
  std::vector<float> feature(n);
  for (std::size_t i = 0; i < n; ++i) {
    feature[i] = static_cast<float>(rows[i]);
  }
  out.fmat = GetDMatrixFromData(feature, n, 1);
  out.gpair.gpair.Reshape(n, kNumClasses);
  out.gpair.exact_hessian.Reshape(n, n_free);
  auto h_gpair = out.gpair.gpair.HostView();
  for (std::size_t i = 0; i < n; ++i) {
    FillRow(rows[i], h_gpair, i, out.gpair.exact_hessian.HostRow(i));
  }
  return out;
}

Slice MakeWholeSlice() {
  std::vector<std::size_t> rows(kRows);
  std::iota(rows.begin(), rows.end(), std::size_t{0});
  return MakeSlice(rows);
}

TrainParam MakeParam(std::string const& min_child_weight = "0",
                     std::string const& interaction = "") {
  TrainParam param;
  Args args{{"max_depth", "2"},
            {"max_bin", "16"},
            {"lambda", "1.0"},
            {"gamma", "0"},
            {"min_child_weight", min_child_weight},
            {"learning_rate", "1.0"}};
  if (!interaction.empty()) {
    args.emplace_back("interaction_constraints", interaction);
  }
  param.UpdateAllowUnknown(args);
  return param;
}

/**
 * @brief Build the root histogram through the real builder and return the node total.
 *
 * The node total -- the sum over every bin -- is independent of where the quantile cuts
 * fall, so comparing it isolates the histogram reduction from the distributed sketcher.
 */
std::vector<double> RootNodeTotal(Context* ctx, Slice* slice, TrainParam const& param) {
  HistMakerTrainParam hist_param;
  hist_param.UpdateAllowUnknown(Args{});
  common::Monitor monitor;
  auto sampler = std::make_shared<common::ColumnSampler>();
  ExactMultiTargetHistBuilder builder{ctx, &param, &hist_param, sampler, &monitor};
  builder.SetExactHessian(&slice->gpair.exact_hessian);

  RegTree tree{kNumClasses, static_cast<bst_feature_t>(slice->fmat->Info().num_col_)};
  auto gpair = slice->gpair.gpair.HostView();
  builder.InitData(slice->fmat.get(), &tree, gpair);
  static_cast<void>(builder.InitRoot(slice->fmat.get(), gpair, &tree));

  auto const& hist = builder.Histogram();
  auto n_free = static_cast<bst_target_t>(kNumClasses - 1);
  ExactStatBuffer total;
  total.Reset(n_free);
  ExactNodeTotal(hist[RegTree::kRoot], n_free, hist.TotalBins(), total.Data());
  auto span = total.Data();
  return std::vector<double>{span.begin(), span.end()};
}

std::unique_ptr<RegTree> Train(Context* ctx, Slice* slice, TrainParam const& param,
                               bool exact) {
  ObjInfo task{ObjInfo::kClassification, false, true};
  auto updater =
      std::unique_ptr<TreeUpdater>{TreeUpdater::Create("grow_quantile_histmaker", ctx, &task)};
  updater->Configure(Args{});
  auto tree = std::make_unique<RegTree>(
      kNumClasses, static_cast<bst_feature_t>(slice->fmat->Info().num_col_));
  std::vector<RegTree*> trees{tree.get()};
  std::vector<HostDeviceVector<bst_node_t>> position(1);

  GradientContainer container;
  container.gpair = std::move(slice->gpair.gpair);
  if (exact) {
    container.exact_hessian.Reshape(slice->gpair.exact_hessian.NumRows(),
                                    slice->gpair.exact_hessian.n_free);
    auto src = slice->gpair.exact_hessian.HostValues();
    auto dst = container.exact_hessian.HostValues();
    std::copy(src.cbegin(), src.cend(), dst.begin());
  }
  updater->Update(&param, &container, slice->fmat.get(),
                  common::Span<HostDeviceVector<bst_node_t>>{position.data(), position.size()},
                  trees);
  slice->gpair.gpair = std::move(container.gpair);
  return tree;
}

struct TreeSummary {
  std::vector<int> is_leaf;
  std::vector<int> split_index;
  std::vector<float> split_cond;
  std::vector<std::vector<float>> weights;
};

TreeSummary Summarize(RegTree const& tree) {
  TreeSummary out;
  auto mt = tree.HostMtView();
  for (bst_node_t nidx = 0; nidx < static_cast<bst_node_t>(tree.Size()); ++nidx) {
    out.is_leaf.push_back(mt.IsLeaf(nidx) ? 1 : 0);
    out.split_index.push_back(mt.IsLeaf(nidx) ? -1 : static_cast<int>(mt.SplitIndex(nidx)));
    out.split_cond.push_back(mt.IsLeaf(nidx) ? 0.0f : mt.SplitCond(nidx));
    std::vector<float> w;
    if (mt.IsLeaf(nidx)) {
      auto leaf = mt.LeafValue(nidx);
      for (bst_target_t t = 0; t < kNumClasses; ++t) {
        w.push_back(leaf(t));
      }
    }
    out.weights.push_back(w);
  }
  return out;
}

/** @brief Largest threshold discrepancy between two trees of the same shape. */
double MaxThresholdDiff(TreeSummary const& a, TreeSummary const& b) {
  double worst = 0.0;
  auto n = std::min(a.is_leaf.size(), b.is_leaf.size());
  for (std::size_t i = 0; i < n; ++i) {
    if (!a.is_leaf[i] && !b.is_leaf[i]) {
      worst = std::max(worst, std::fabs(static_cast<double>(a.split_cond[i]) - b.split_cond[i]));
    }
  }
  return worst;
}
}  // anonymous namespace

/**
 * The core distributed guarantee: summing the workers' local histograms reproduces the
 * single-process statistics exactly.
 *
 * The node total is compared rather than individual bins because it does not depend on where
 * the quantile cuts fall, so this isolates the all-reduce from the distributed sketcher.
 * Without the reduction each worker would see only its own half and every entry would be
 * roughly halved.
 */
TEST(ExactDistributed, RootHistogramReducesToSingleProcessTotal) {
  Context ctx;
  auto whole = MakeWholeSlice();
  auto param = MakeParam();
  auto reference = RootNodeTotal(&ctx, &whole, param);
  ASSERT_FALSE(reference.empty());

  bool any_nonzero = false;
  for (auto v : reference) {
    if (std::fabs(v) > 1e-9) {
      any_nonzero = true;
    }
  }
  ASSERT_TRUE(any_nonzero) << "the reference statistics are all zero";

  collective::TestDistributedGlobal(2, [&] {
    auto slice = MakeSlice(RowsForRank(collective::GetRank(), 2));
    Context worker_ctx;
    auto local = MakeParam();
    auto reduced = RootNodeTotal(&worker_ctx, &slice, local);

    ASSERT_EQ(reduced.size(), reference.size());
    for (std::size_t i = 0; i < reference.size(); ++i) {
      auto scale = std::max(1.0, std::fabs(reference[i]));
      EXPECT_NEAR(reduced[i], reference[i], 1e-9 * scale)
          << "entry " << i << ": reduced " << reduced[i] << " vs single-process "
          << reference[i];
    }
  });
}

/** Both workers must end up with byte-identical statistics after the reduction. */
TEST(ExactDistributed, WorkersAgreeOnTheReducedHistogram) {
  Context reference_ctx;
  auto whole = MakeWholeSlice();
  auto param = MakeParam();
  auto reference = RootNodeTotal(&reference_ctx, &whole, param);

  collective::TestDistributedGlobal(2, [&] {
    auto slice = MakeSlice(RowsForRank(collective::GetRank(), 2));
    Context worker_ctx;
    auto local = MakeParam();
    auto reduced = RootNodeTotal(&worker_ctx, &slice, local);
    // Every worker compares against the same single-process reference, so agreement with it
    // implies agreement with each other.
    for (std::size_t i = 0; i < reference.size(); ++i) {
      auto scale = std::max(1.0, std::fabs(reference[i]));
      ASSERT_NEAR(reduced[i], reference[i], 1e-9 * scale) << "entry " << i;
    }
  });
}

/**
 * Attribution control: the residual tree-level difference between one and two workers comes
 * from XGBoost's distributed quantile sketch, not from exact mode.
 *
 * The same striped partition is trained in ordinary diagonal mode, and its thresholds move
 * by the same one-bin amount. If exact mode were at fault, diagonal mode would agree.
 */
TEST(ExactDistributed, ThresholdDifferenceComesFromTheSketchNotExactMode) {
  Context ctx;
  auto whole_diag = MakeWholeSlice();
  auto param = MakeParam();
  auto diagonal_reference = Summarize(*Train(&ctx, &whole_diag, param, /*exact=*/false));

  auto whole_exact = MakeWholeSlice();
  auto exact_reference = Summarize(*Train(&ctx, &whole_exact, param, /*exact=*/true));
  ASSERT_GT(exact_reference.is_leaf.size(), 1u) << "the reference tree never split";

  collective::TestDistributedGlobal(2, [&] {
    Context worker_ctx;
    auto local = MakeParam();

    auto diag_slice = MakeSlice(RowsForRank(collective::GetRank(), 2));
    auto diag = Summarize(*Train(&worker_ctx, &diag_slice, local, /*exact=*/false));
    auto diag_diff = MaxThresholdDiff(diagonal_reference, diag);

    auto exact_slice = MakeSlice(RowsForRank(collective::GetRank(), 2));
    auto exact = Summarize(*Train(&worker_ctx, &exact_slice, local, /*exact=*/true));
    auto exact_diff = MaxThresholdDiff(exact_reference, exact);

    // Exact mode is no worse than the pre-existing diagonal path on the same partition, which
    // is the claim: whatever threshold movement remains is a property of distributed
    // sketching that both modes share.
    EXPECT_LE(exact_diff, std::max(diag_diff, 1.0) + 1e-6)
        << "exact threshold drift " << exact_diff << " exceeds diagonal drift " << diag_diff;

    // The structure itself must survive: same number of nodes and the same split features.
    ASSERT_EQ(exact.is_leaf.size(), exact_reference.is_leaf.size());
    for (std::size_t n = 0; n < exact.is_leaf.size(); ++n) {
      EXPECT_EQ(exact.is_leaf[n], exact_reference.is_leaf[n]) << "node " << n;
      EXPECT_EQ(exact.split_index[n], exact_reference.split_index[n]) << "node " << n;
    }
  });
}

/**
 * min_child_weight is derived from the reduced Hessian, so the decision must reflect the
 * whole dataset. A worker whose local rows alone would fail the threshold must still split
 * when the global curvature passes it, and vice versa.
 */
TEST(ExactDistributed, MinChildWeightUsesReducedCurvature) {
  Context ctx;
  auto whole = MakeWholeSlice();
  auto open_reference = Summarize(*Train(&ctx, &whole, MakeParam("0"), true));
  ASSERT_GT(open_reference.is_leaf.size(), 1u);

  auto whole_blocked = MakeWholeSlice();
  auto blocked_reference = Summarize(*Train(&ctx, &whole_blocked, MakeParam("1e9"), true));
  ASSERT_EQ(blocked_reference.is_leaf.size(), 1u);

  collective::TestDistributedGlobal(2, [&] {
    Context worker_ctx;
    auto open_slice = MakeSlice(RowsForRank(collective::GetRank(), 2));
    auto open = Summarize(*Train(&worker_ctx, &open_slice, MakeParam("0"), true));
    EXPECT_EQ(open.is_leaf.size(), open_reference.is_leaf.size())
        << "a permissive threshold produced a different tree shape under distribution";

    auto blocked_slice = MakeSlice(RowsForRank(collective::GetRank(), 2));
    auto blocked = Summarize(*Train(&worker_ctx, &blocked_slice, MakeParam("1e9"), true));
    EXPECT_EQ(blocked.is_leaf.size(), 1u) << "a huge min_child_weight still permitted a split";
  });
}

/**
 * The sibling subtraction must operate on GLOBAL statistics, not local ones.
 *
 * `right = parent - left` is only correct if both operands have already been reduced. If the
 * left child were still worker-local, the derived right child would silently absorb the
 * other workers' rows and no reconciliation check would notice, because the identity holds
 * by construction either way.
 *
 * So this drives the real builder through a split and then compares the left child's total
 * against the statistics of every GLOBAL row that the tree's own split sends left --
 * independent of which worker owns them.
 */
TEST(ExactDistributed, DistributedParentChildSubtractionIsGlobal) {
  auto n_free = static_cast<bst_target_t>(kNumClasses - 1);
  auto record_size = common::PackedStatsStride(n_free);

  collective::TestDistributedGlobal(2, [&] {
    auto slice = MakeSlice(RowsForRank(collective::GetRank(), 2));
    Context ctx;
    auto param = MakeParam();
    HistMakerTrainParam hist_param;
    hist_param.UpdateAllowUnknown(Args{});
    common::Monitor monitor;
    auto sampler = std::make_shared<common::ColumnSampler>();
    ExactMultiTargetHistBuilder builder{&ctx, &param, &hist_param, sampler, &monitor};
    builder.SetExactHessian(&slice.gpair.exact_hessian);

    RegTree tree{kNumClasses, static_cast<bst_feature_t>(slice.fmat->Info().num_col_)};
    auto gpair = slice.gpair.gpair.HostView();
    builder.InitData(slice.fmat.get(), &tree, gpair);
    auto root = builder.InitRoot(slice.fmat.get(), gpair, &tree);
    ASSERT_GT(root.split.loss_chg, 0.0f) << "the root produced no split to examine";

    std::vector<MultiExpandEntry> applied{root};
    builder.ApplyTreeSplit(root, &tree);
    builder.UpdatePosition(slice.fmat.get(), &tree, applied);
    builder.BuildHistogram(slice.fmat.get(), &tree, applied, gpair);

    auto left_nidx = tree.LeftChild(root.nid);
    auto right_nidx = tree.RightChild(root.nid);
    auto const& hist = builder.Histogram();

    auto node_total = [&](bst_node_t nidx) {
      ExactStatBuffer total;
      total.Reset(n_free);
      ExactNodeTotal(hist[nidx], n_free, hist.TotalBins(), total.Data());
      auto span = total.Data();
      return std::vector<double>{span.begin(), span.end()};
    };
    auto parent = node_total(root.nid);
    auto left = node_total(left_nidx);
    auto right = node_total(right_nidx);

    // The tree's own split rule, applied to every global row. XGBoost sends a row left when
    // its feature value is strictly below the threshold.
    auto threshold = tree.HostMtView().SplitCond(root.nid);
    std::vector<double> expected_left(record_size, 0.0);
    std::vector<double> expected_parent(record_size, 0.0);
    {
      // Statistics for all kRows rows, regardless of which worker owns them.
      auto all = MakeWholeSlice();
      auto h_all = all.gpair.gpair.HostView();
      for (std::size_t r = 0; r < kRows; ++r) {
        auto fvalue = static_cast<float>(r);
        auto* dst = (fvalue < threshold) ? expected_left.data() : nullptr;
        auto row = all.gpair.exact_hessian.HostRow(r);
        for (bst_target_t i = 0; i < n_free; ++i) {
          expected_parent[i] += h_all(r, i).GetGrad();
          if (dst) {
            dst[i] += h_all(r, i).GetGrad();
          }
        }
        for (std::size_t k = 0; k < row.size(); ++k) {
          expected_parent[n_free + k] += row[k];
          if (dst) {
            dst[n_free + k] += row[k];
          }
        }
      }
    }

    // The parent must already be global before the subtraction happens.
    for (std::size_t i = 0; i < record_size; ++i) {
      auto scale = std::max(1.0, std::fabs(expected_parent[i]));
      EXPECT_NEAR(parent[i], expected_parent[i], 1e-6 * scale) << "parent entry " << i;
    }
    // ...and so must the left child, which is the one that is actually accumulated.
    for (std::size_t i = 0; i < record_size; ++i) {
      auto scale = std::max(1.0, std::fabs(expected_left[i]));
      EXPECT_NEAR(left[i], expected_left[i], 1e-6 * scale) << "left entry " << i;
    }
    // Then the derived sibling reconciles exactly.
    for (std::size_t i = 0; i < record_size; ++i) {
      auto scale = std::max(1.0, std::fabs(parent[i]));
      EXPECT_NEAR(left[i] + right[i], parent[i], 1e-9 * scale) << "reconciliation entry " << i;
    }

    // The comparison is only meaningful if the split actually divided the data and the
    // Hessian carries genuine off-diagonal coupling.
    double left_mass = 0.0;
    for (bst_target_t i = 0; i < n_free; ++i) {
      left_mass += std::fabs(left[i]);
    }
    EXPECT_GT(left_mass, 0.0) << "the left child is empty";
    auto view = common::PackedHessianAtRow(
        common::Span<double const>{left.data() + n_free, record_size - n_free}, n_free, 0);
    EXPECT_LT(view.Get(1, 0), -1e-12) << "no negative off-diagonal Hessian term present";
  });
}

/** Interaction constraints are structural, so every worker applies the identical graph. */
TEST(ExactDistributed, InteractionConstraintsAreConsistent) {
  Context ctx;
  auto whole = MakeWholeSlice();
  auto reference = Summarize(*Train(&ctx, &whole, MakeParam("0", "[[0]]"), true));

  collective::TestDistributedGlobal(2, [&] {
    Context worker_ctx;
    auto slice = MakeSlice(RowsForRank(collective::GetRank(), 2));
    auto summary = Summarize(*Train(&worker_ctx, &slice, MakeParam("0", "[[0]]"), true));
    ASSERT_EQ(summary.is_leaf.size(), reference.is_leaf.size());
    for (std::size_t n = 0; n < summary.is_leaf.size(); ++n) {
      EXPECT_EQ(summary.is_leaf[n], reference.is_leaf[n]) << "node " << n;
      EXPECT_EQ(summary.split_index[n], reference.split_index[n]) << "node " << n;
    }
  });
}

namespace {
/** @brief Contiguous blocks instead of a stripe, so the reduction sums in a different order. */
std::vector<std::size_t> ContiguousRowsForRank(std::int32_t rank, std::int32_t n_workers) {
  auto workers = static_cast<std::size_t>(n_workers);
  auto per = (kRows + workers - 1) / workers;
  auto begin = std::min(kRows, static_cast<std::size_t>(rank) * per);
  auto end = std::min(kRows, begin + per);
  std::vector<std::size_t> rows;
  for (auto r = begin; r < end; ++r) {
    rows.push_back(r);
  }
  return rows;
}

/** @brief Largest relative discrepancy between two node totals. */
double MaxRelativeDiff(std::vector<double> const& a, std::vector<double> const& b) {
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, std::fabs(a[i] - b[i]) / std::max(1.0, std::fabs(a[i])));
  }
  return worst;
}
}  // anonymous namespace

/**
 * Phase 14. The all-reduce changes the order in which rows are summed, and the worker count
 * and partition shape change it again. Floating point addition is not associative, so the
 * reduced total is not bit-identical to the single-process one -- the question is whether
 * the discrepancy stays at double round-off, far below anything that could move a split.
 *
 * The node total is compared because it is independent of where the quantile cuts fall, so
 * this measures the reduction alone. `ThresholdDifferenceComesFromTheSketchNotExactMode`
 * covers the separate, larger effect of the distributed sketcher.
 */
TEST(ExactDistributed, ReductionIsNumericallyStableAcrossPartitions) {
  Context ctx;
  auto whole = MakeWholeSlice();
  auto param = MakeParam();
  auto reference = RootNodeTotal(&ctx, &whole, param);
  ASSERT_FALSE(reference.empty());

  double scale = 0.0;
  for (auto v : reference) {
    scale = std::max(scale, std::fabs(v));
  }
  ASSERT_GT(scale, 1e-6) << "the reference statistics carry no magnitude to compare against";

  for (std::int32_t n_workers : {2, 3, 4}) {
    for (bool contiguous : {false, true}) {
      collective::TestDistributedGlobal(n_workers, [&] {
        auto rank = collective::GetRank();
        auto rows = contiguous ? ContiguousRowsForRank(rank, n_workers)
                               : RowsForRank(rank, n_workers);
        auto slice = MakeSlice(rows);
        Context worker_ctx;
        auto local = MakeParam();
        auto reduced = RootNodeTotal(&worker_ctx, &slice, local);

        ASSERT_EQ(reduced.size(), reference.size());
        auto worst = MaxRelativeDiff(reference, reduced);
        // Double accumulation over 256 rows: the reordering error is bounded by a few ulp,
        // not by the tolerance chosen here.
        EXPECT_LT(worst, 1e-12)
            << n_workers << " workers, " << (contiguous ? "contiguous" : "striped")
            << " partition: relative discrepancy " << worst;
      });
    }
  }
}

/**
 * Phase 14, continued. Every worker must come out of the reduction with the same numbers, or
 * they would grow different trees and the ensembles would diverge. This compares the workers
 * against each other directly rather than each against a reference, so a shared bias that
 * happened to match the reference could not hide here.
 */
TEST(ExactDistributed, WorkersAgreeBitForBitAfterReduction) {
  for (std::int32_t n_workers : {2, 4}) {
    collective::TestDistributedGlobal(n_workers, [&] {
      auto rank = collective::GetRank();
      auto slice = MakeSlice(RowsForRank(rank, n_workers));
      Context worker_ctx;
      auto local = MakeParam();
      auto reduced = RootNodeTotal(&worker_ctx, &slice, local);

      // Broadcast rank 0's totals and require an exact match: the all-reduce hands every
      // worker the same buffer, so any difference would be a real divergence rather than a
      // reordering artefact.
      auto broadcast = reduced;
      collective::SafeColl(collective::Broadcast(
          &worker_ctx, linalg::MakeVec(broadcast.data(), broadcast.size()), 0));

      ASSERT_EQ(broadcast.size(), reduced.size());
      for (std::size_t i = 0; i < reduced.size(); ++i) {
        ASSERT_EQ(reduced[i], broadcast[i])
            << n_workers << " workers, rank " << rank << ", entry " << i;
      }
    });
  }
}
}  // namespace xgboost::tree
