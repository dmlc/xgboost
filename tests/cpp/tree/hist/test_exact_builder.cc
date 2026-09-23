/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>
#include <xgboost/context.h>
#include <xgboost/gradient.h>
#include <xgboost/json.h>
#include <xgboost/tree_model.h>
#include <xgboost/tree_updater.h>

#include <cmath>    // for fabs, exp
#include <cstddef>  // for size_t
#include <memory>   // for unique_ptr
#include <numeric>  // for iota
#include <vector>   // for vector

#include "../../../../src/common/exact_multinomial/packed_stats.h"
#include "../../../../src/tree/hist/exact_builder.h"
#include "../../filesystem.h"  // for TemporaryDirectory
#include "../../helpers.h"

namespace xgboost::tree {
namespace {
/**
 * @brief A deterministic multinomial problem whose single feature separates the classes.
 *
 * Statistics are produced with the same formulas the objective uses, so the updater sees
 * exactly what live training would hand it.
 */
struct ExactProblem {
  std::shared_ptr<DMatrix> fmat;
  GradientContainer gpair;
  std::size_t n_rows;
  bst_target_t n_classes;
};

ExactProblem MakeProblem(std::size_t n_rows, bst_target_t n_classes, double signal = 3.0) {
  ExactProblem out;
  out.n_rows = n_rows;
  out.n_classes = n_classes;
  auto n_free = static_cast<bst_target_t>(n_classes - 1);

  std::vector<float> feature(n_rows);
  std::iota(feature.begin(), feature.end(), 0.0f);
  out.fmat = GetDMatrixFromData(feature, n_rows, 1);

  out.gpair.gpair.Reshape(n_rows, n_classes);
  out.gpair.exact_hessian.Reshape(n_rows, n_free);
  auto h_gpair = out.gpair.gpair.HostView();

  for (std::size_t r = 0; r < n_rows; ++r) {
    bool upper = r >= n_rows / 2;
    // Two regimes, so a split on the ramp feature is informative.
    std::vector<double> p(n_classes, 1.0);
    p[upper ? (n_classes - 1) : 0] += signal;
    double total = 0.0;
    for (auto v : p) {
      total += v;
    }
    for (auto& v : p) {
      v /= total;
    }
    std::size_t label = upper ? 1 : 0;

    for (bst_target_t t = 0; t < n_classes; ++t) {
      auto grad = static_cast<float>(p[t] - (label == t ? 1.0 : 0.0));
      h_gpair(r, t) = GradientPair{grad, std::max(std::fabs(grad), 1e-16f)};
    }
    auto row = out.gpair.exact_hessian.HostRow(r);
    auto view = common::PackedHessianAtRow(row, n_free, 0);
    for (std::size_t i = 0; i < n_free; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        view.Set(i, j, static_cast<float>(p[i] * ((i == j ? 1.0 : 0.0) - p[j])));
      }
    }
  }
  return out;
}

TrainParam MakeParam(std::string const& max_depth = "2", std::string const& lambda = "1.0",
                     std::string const& subsample = "1.0") {
  TrainParam param;
  param.UpdateAllowUnknown(Args{{"max_depth", max_depth},
                                {"max_bin", "16"},
                                {"lambda", lambda},
                                {"gamma", "0"},
                                {"min_child_weight", "0"},
                                {"subsample", subsample},
                                {"learning_rate", "1.0"}});
  return param;
}

/** @brief Run one round of exact training and return the grown tree. */
std::unique_ptr<RegTree> TrainExact(Context* ctx, ExactProblem* problem, TrainParam const& param) {
  ObjInfo task{ObjInfo::kClassification, false, true};
  auto updater =
      std::unique_ptr<TreeUpdater>{TreeUpdater::Create("grow_quantile_histmaker", ctx, &task)};
  updater->Configure(Args{});

  auto tree = std::make_unique<RegTree>(problem->n_classes,
                                        static_cast<bst_feature_t>(problem->fmat->Info().num_col_));
  std::vector<RegTree*> trees{tree.get()};
  std::vector<HostDeviceVector<bst_node_t>> position(1);
  updater->Update(&param, &problem->gpair, problem->fmat.get(),
                  common::Span<HostDeviceVector<bst_node_t>>{position.data(), position.size()},
                  trees);
  return tree;
}

/** @brief Every leaf's K outputs, read back from the grown tree. */
std::vector<std::vector<float>> LeafWeights(RegTree const& tree, bst_target_t n_classes) {
  std::vector<std::vector<float>> out;
  auto mt = tree.HostMtView();
  for (bst_node_t nidx = 0; nidx < static_cast<bst_node_t>(tree.Size()); ++nidx) {
    if (!mt.IsLeaf(nidx)) {
      continue;
    }
    auto leaf = mt.LeafValue(nidx);
    std::vector<float> values;
    for (bst_target_t t = 0; t < n_classes; ++t) {
      values.push_back(leaf(t));
    }
    out.push_back(values);
  }
  return out;
}
}  // anonymous namespace

/** The updater must actually take the exact branch and grow a shared vector-leaf tree. */
TEST(ExactBuilder, GrowsSharedVectorTree) {
  for (bst_target_t n_classes : {2u, 3u, 7u}) {
    Context ctx;
    ctx.InitAllowUnknown(Args{{"nthread", "2"}});
    auto problem = MakeProblem(256, n_classes);
    auto param = MakeParam();
    auto tree = TrainExact(&ctx, &problem, param);

    ASSERT_TRUE(tree->IsMultiTarget()) << "K=" << n_classes;
    EXPECT_EQ(tree->NumTargets(), n_classes) << "K=" << n_classes;
    // One shared structure, not K separate trees.
    EXPECT_GT(tree->Size(), 1u) << "K=" << n_classes << ": the tree never split";

    auto leaves = LeafWeights(*tree, n_classes);
    ASSERT_FALSE(leaves.empty()) << "K=" << n_classes;
    for (auto const& leaf : leaves) {
      ASSERT_EQ(leaf.size(), n_classes) << "K=" << n_classes;
      for (auto v : leaf) {
        EXPECT_TRUE(std::isfinite(v)) << "K=" << n_classes << ": non-finite leaf output";
      }
    }
  }
}

/**
 * The K-1 -> K write must be centered: each leaf's outputs sum to zero.
 *
 * This is the gauge the regularizer was derived for, so a violation means the leaf is being
 * penalised differently from how it was solved.
 */
TEST(ExactBuilder, LeafOutputsAreCentered) {
  for (bst_target_t n_classes : {2u, 3u, 7u}) {
    Context ctx;
    auto problem = MakeProblem(256, n_classes);
    auto param = MakeParam();
    auto tree = TrainExact(&ctx, &problem, param);

    auto leaves = LeafWeights(*tree, n_classes);
    ASSERT_FALSE(leaves.empty());
    bool any_nonzero = false;
    for (auto const& leaf : leaves) {
      double total = 0.0;
      for (auto v : leaf) {
        total += v;
        if (std::fabs(v) > 1e-6) {
          any_nonzero = true;
        }
      }
      EXPECT_NEAR(total, 0.0, 1e-4) << "K=" << n_classes << ": leaf outputs are not centered";
    }
    // A tree of all-zero leaves would satisfy centering vacuously.
    EXPECT_TRUE(any_nonzero) << "K=" << n_classes << ": every leaf output was zero";
  }
}

/** The unit-level centering conversion, checked directly. */
TEST(ExactBuilder, CenteredLeafWeightConversion) {
  // w = [1, 2, 3] with a reference class gives delta = [1, 2, 3, 0], mean 1.5.
  std::vector<double> free_weight{1.0, 2.0, 3.0};
  std::vector<float> out(4, 0.0f);
  CenteredLeafWeight(common::Span<double const>{free_weight.data(), free_weight.size()},
                     common::Span<float>{out.data(), out.size()});
  EXPECT_NEAR(out[0], -0.5f, 1e-6);
  EXPECT_NEAR(out[1], 0.5f, 1e-6);
  EXPECT_NEAR(out[2], 1.5f, 1e-6);
  EXPECT_NEAR(out[3], -1.5f, 1e-6);
  double total = 0.0;
  for (auto v : out) {
    total += v;
  }
  EXPECT_NEAR(total, 0.0, 1e-6);

  // Softmax invariance: centering must not change the implied probabilities relative to the
  // raw embedding [w, 0].
  std::vector<double> raw{1.0, 2.0, 3.0, 0.0};
  auto softmax = [](std::vector<double> z) {
    double m = *std::max_element(z.begin(), z.end());
    double s = 0.0;
    for (auto& v : z) {
      v = std::exp(v - m);
      s += v;
    }
    for (auto& v : z) {
      v /= s;
    }
    return z;
  };
  auto p_raw = softmax(raw);
  auto p_centered = softmax(std::vector<double>{out.begin(), out.end()});
  for (std::size_t k = 0; k < 4; ++k) {
    EXPECT_NEAR(p_raw[k], p_centered[k], 1e-9) << "class " << k;
  }
}

/**
 * Off-diagonal Hessian terms must change the trained tree.
 *
 * Without this, exact mode could silently be a diagonal approximation and every structural
 * test above would still pass.
 */
TEST(ExactBuilder, OffDiagonalTermsChangeTraining) {
  bst_target_t constexpr kNumClasses = 4;
  Context ctx;
  auto full = MakeProblem(256, kNumClasses);
  auto diagonal = MakeProblem(256, kNumClasses);

  // Strip the off-diagonal Hessian entries from the second problem only.
  auto n_free = static_cast<bst_target_t>(kNumClasses - 1);
  for (std::size_t r = 0; r < diagonal.n_rows; ++r) {
    auto row = diagonal.gpair.exact_hessian.HostRow(r);
    auto view = common::PackedHessianAtRow(row, n_free, 0);
    for (std::size_t i = 0; i < n_free; ++i) {
      for (std::size_t j = 0; j < i; ++j) {
        view.Set(i, j, 0.0f);
      }
    }
  }

  auto param = MakeParam();
  auto tree_full = TrainExact(&ctx, &full, param);
  auto tree_diag = TrainExact(&ctx, &diagonal, param);

  auto leaves_full = LeafWeights(*tree_full, kNumClasses);
  auto leaves_diag = LeafWeights(*tree_diag, kNumClasses);
  ASSERT_FALSE(leaves_full.empty());
  ASSERT_EQ(leaves_full.size(), leaves_diag.size());

  bool differs = false;
  for (std::size_t l = 0; l < leaves_full.size() && !differs; ++l) {
    for (bst_target_t t = 0; t < kNumClasses; ++t) {
      if (std::fabs(leaves_full[l][t] - leaves_diag[l][t]) > 1e-5) {
        differs = true;
        break;
      }
    }
  }
  EXPECT_TRUE(differs)
      << "dropping the off-diagonal Hessian left the trained leaves unchanged, so the dense "
         "Hessian is not reaching live training";
}

/** Sampling keeps gradient and Hessian aligned inside live training. */
TEST(ExactBuilder, TrainsUnderSubsampling) {
  bst_target_t constexpr kNumClasses = 3;
  for (double subsample : {0.3, 0.5, 0.8}) {
    Context ctx;
    auto problem = MakeProblem(512, kNumClasses);
    auto param = MakeParam("2", "1.0", std::to_string(subsample));
    auto tree = TrainExact(&ctx, &problem, param);

    ASSERT_TRUE(tree->IsMultiTarget());
    auto leaves = LeafWeights(*tree, kNumClasses);
    ASSERT_FALSE(leaves.empty()) << "subsample=" << subsample;
    for (auto const& leaf : leaves) {
      double total = 0.0;
      for (auto v : leaf) {
        EXPECT_TRUE(std::isfinite(v)) << "subsample=" << subsample;
        total += v;
      }
      EXPECT_NEAR(total, 0.0, 1e-4) << "subsample=" << subsample;
    }
    // The updater samples a copy of both the gradient and the sidecar, so the caller's
    // container must come back untouched. Masking the sidecar in place would compound
    // across trees and misalign it from each tree's own freshly drawn row mask.
    auto h_gpair = problem.gpair.gpair.HostView();
    for (std::size_t r = 0; r < problem.n_rows; ++r) {
      bool gradient_zero = true;
      for (bst_target_t t = 0; t < kNumClasses; ++t) {
        if (h_gpair(r, t).GetGrad() != 0.0f || h_gpair(r, t).GetHess() != 0.0f) {
          gradient_zero = false;
          break;
        }
      }
      bool hessian_zero = true;
      for (auto v : problem.gpair.exact_hessian.HostRow(r)) {
        if (v != 0.0f) {
          hessian_zero = false;
          break;
        }
      }
      ASSERT_FALSE(gradient_zero) << "row " << r << ": the caller's gradient was mutated";
      ASSERT_FALSE(hessian_zero) << "row " << r << ": the caller's sidecar was mutated";
    }
  }
}

/**
 * Two trees in one round must each get their own row mask.
 *
 * The sidecar is copied per tree for exactly this reason; reusing a masked sidecar would
 * drop rows cumulatively and silently shrink the second tree's data.
 */
TEST(ExactBuilder, MultipleTreesPerRoundStayAligned) {
  bst_target_t constexpr kNumClasses = 3;
  Context ctx;
  auto problem = MakeProblem(512, kNumClasses);
  auto param = MakeParam("2", "1.0", "0.5");

  ObjInfo task{ObjInfo::kClassification, false, true};
  auto updater =
      std::unique_ptr<TreeUpdater>{TreeUpdater::Create("grow_quantile_histmaker", &ctx, &task)};
  updater->Configure(Args{});

  auto n_features = static_cast<bst_feature_t>(problem.fmat->Info().num_col_);
  RegTree tree_a{kNumClasses, n_features};
  RegTree tree_b{kNumClasses, n_features};
  std::vector<RegTree*> trees{&tree_a, &tree_b};
  std::vector<HostDeviceVector<bst_node_t>> position(2);
  updater->Update(&param, &problem.gpair, problem.fmat.get(),
                  common::Span<HostDeviceVector<bst_node_t>>{position.data(), position.size()},
                  trees);

  for (auto const* tree : {&tree_a, &tree_b}) {
    ASSERT_TRUE(tree->IsMultiTarget());
    auto leaves = LeafWeights(*tree, kNumClasses);
    ASSERT_FALSE(leaves.empty());
    for (auto const& leaf : leaves) {
      double total = 0.0;
      for (auto v : leaf) {
        EXPECT_TRUE(std::isfinite(v));
        total += v;
      }
      EXPECT_NEAR(total, 0.0, 1e-4);
    }
  }
  // Neither tree may have consumed the caller's sidecar.
  for (std::size_t r = 0; r < problem.n_rows; ++r) {
    bool hessian_zero = true;
    for (auto v : problem.gpair.exact_hessian.HostRow(r)) {
      if (v != 0.0f) {
        hessian_zero = false;
        break;
      }
    }
    ASSERT_FALSE(hessian_zero) << "row " << r << ": sidecar consumed by an earlier tree";
  }

  // The two trees are drawn with subsample=0.5 from independent Bernoulli sequences, so they
  // must not be identical. Checking only that both are finite and centered would pass even if
  // the second tree had silently reused the first tree's sample -- which is exactly the bug
  // the per-tree `exact_sample_` copy exists to prevent.
  auto describe = [&](RegTree const& tree) {
    std::string out;
    auto mt = tree.HostMtView();
    for (bst_node_t nidx = 0; nidx < static_cast<bst_node_t>(tree.Size()); ++nidx) {
      if (mt.IsLeaf(nidx)) {
        out += "L";
        auto w = mt.LeafValue(nidx);
        for (bst_target_t k = 0; k < kNumClasses; ++k) {
          out += " " + std::to_string(w(k));
        }
      } else {
        out += "S " + std::to_string(mt.SplitIndex(nidx)) + " " +
               std::to_string(mt.SplitCond(nidx));
      }
      out += ";";
    }
    return out;
  };
  EXPECT_NE(describe(tree_a), describe(tree_b))
      << "both trees in the round are identical, so the per-tree sampling is not independent";
}

/** gamma suppresses expansion in live training. */
TEST(ExactBuilder, GammaSuppressesExpansion) {
  bst_target_t constexpr kNumClasses = 3;
  Context ctx;
  auto permissive = MakeProblem(256, kNumClasses);
  auto tree_open = TrainExact(&ctx, &permissive, MakeParam());
  ASSERT_GT(tree_open->Size(), 1u);

  auto blocked_problem = MakeProblem(256, kNumClasses);
  TrainParam blocked;
  blocked.UpdateAllowUnknown(Args{{"max_depth", "2"},
                                  {"max_bin", "16"},
                                  {"lambda", "1.0"},
                                  {"gamma", "1e9"},
                                  {"min_child_weight", "0"},
                                  {"learning_rate", "1.0"}});
  auto tree_blocked = TrainExact(&ctx, &blocked_problem, blocked);
  EXPECT_EQ(tree_blocked->Size(), 1u) << "a huge gamma still permitted a split";
}

/** min_child_weight suppresses expansion in live training. */
TEST(ExactBuilder, MinChildWeightSuppressesExpansion) {
  bst_target_t constexpr kNumClasses = 3;
  Context ctx;
  auto problem = MakeProblem(256, kNumClasses);
  TrainParam strict;
  strict.UpdateAllowUnknown(Args{{"max_depth", "2"},
                                 {"max_bin", "16"},
                                 {"lambda", "1.0"},
                                 {"gamma", "0"},
                                 {"min_child_weight", "1e9"},
                                 {"learning_rate", "1.0"}});
  auto tree = TrainExact(&ctx, &problem, strict);
  EXPECT_EQ(tree->Size(), 1u) << "a huge min_child_weight still permitted a split";
}

/** Exact mode rejects configurations it cannot honour rather than approximating them. */
TEST(ExactBuilder, RejectsUnsupportedConfigurations) {
  bst_target_t constexpr kNumClasses = 3;
  Context ctx;

  // reg_alpha has no closed form for a coupled system.
  {
    auto problem = MakeProblem(64, kNumClasses);
    TrainParam param;
    param.UpdateAllowUnknown(
        Args{{"max_depth", "1"}, {"max_bin", "16"}, {"alpha", "0.5"}, {"learning_rate", "1.0"}});
    EXPECT_THROW(TrainExact(&ctx, &problem, param), dmlc::Error);
  }
  // A scalar tree cannot carry a joint solve.
  {
    auto problem = MakeProblem(64, kNumClasses);
    ObjInfo task{ObjInfo::kClassification, false, true};
    auto updater =
        std::unique_ptr<TreeUpdater>{TreeUpdater::Create("grow_quantile_histmaker", &ctx, &task)};
    updater->Configure(Args{});
    auto param = MakeParam("1");
    RegTree scalar_tree{1, static_cast<bst_feature_t>(problem.fmat->Info().num_col_)};
    ASSERT_FALSE(scalar_tree.IsMultiTarget());
    std::vector<RegTree*> trees{&scalar_tree};
    std::vector<HostDeviceVector<bst_node_t>> position(1);
    EXPECT_THROW(
        updater->Update(
            &param, &problem.gpair, problem.fmat.get(),
            common::Span<HostDeviceVector<bst_node_t>>{position.data(), position.size()}, trees),
        dmlc::Error);
  }
}

/**
 * The grown vector-leaf tree must round-trip through the existing serializer unchanged.
 *
 * Exact mode writes ordinary K-output vector leaves, so no format change is involved; this
 * pins that down. Training-time Hessians are never serialized -- only the final weights.
 */
TEST(ExactBuilder, SerializationRoundTrip) {
  for (bst_target_t n_classes : {2u, 3u, 7u}) {
    Context ctx;
    auto problem = MakeProblem(256, n_classes);
    auto param = MakeParam();
    auto tree = TrainExact(&ctx, &problem, param);
    ASSERT_GT(tree->Size(), 1u) << "K=" << n_classes;

    Json saved{Object{}};
    tree->SaveModel(&saved);

    RegTree restored;
    restored.LoadModel(saved);

    ASSERT_TRUE(restored.IsMultiTarget()) << "K=" << n_classes;
    ASSERT_EQ(restored.NumTargets(), n_classes) << "K=" << n_classes;
    ASSERT_EQ(restored.Size(), tree->Size()) << "K=" << n_classes;

    auto before = LeafWeights(*tree, n_classes);
    auto after = LeafWeights(restored, n_classes);
    ASSERT_EQ(before.size(), after.size()) << "K=" << n_classes;
    ASSERT_FALSE(before.empty());

    for (std::size_t l = 0; l < before.size(); ++l) {
      for (bst_target_t t = 0; t < n_classes; ++t) {
        EXPECT_FLOAT_EQ(before[l][t], after[l][t])
            << "K=" << n_classes << " leaf " << l << " target " << t;
      }
    }

    // Structure survives too: same splits, same directions.
    auto mt_before = tree->HostMtView();
    auto mt_after = restored.HostMtView();
    for (bst_node_t nidx = 0; nidx < static_cast<bst_node_t>(tree->Size()); ++nidx) {
      ASSERT_EQ(mt_before.IsLeaf(nidx), mt_after.IsLeaf(nidx)) << "node " << nidx;
      if (!mt_before.IsLeaf(nidx)) {
        EXPECT_EQ(mt_before.SplitIndex(nidx), mt_after.SplitIndex(nidx)) << "node " << nidx;
        EXPECT_FLOAT_EQ(mt_before.SplitCond(nidx), mt_after.SplitCond(nidx)) << "node " << nidx;
      }
    }

    // Centering is preserved by the round trip.
    for (auto const& leaf : after) {
      double total = 0.0;
      for (auto v : leaf) {
        total += v;
      }
      EXPECT_NEAR(total, 0.0, 1e-4) << "K=" << n_classes;
    }
  }
}

/** Without a sidecar the updater must take the untouched existing path. */
TEST(ExactBuilder, NoSidecarUsesExistingPath) {
  bst_target_t constexpr kNumClasses = 3;
  Context ctx;
  auto problem = MakeProblem(256, kNumClasses);
  // Drop the sidecar: this is ordinary multi-target training.
  problem.gpair.ClearExactHessian();
  ASSERT_FALSE(problem.gpair.HasExactHessian());

  auto param = MakeParam();
  auto tree = TrainExact(&ctx, &problem, param);
  ASSERT_TRUE(tree->IsMultiTarget());
  // The existing per-target path does not center its leaves, so a centered result here would
  // mean the exact branch had been taken by mistake.
  auto leaves = LeafWeights(*tree, kNumClasses);
  ASSERT_FALSE(leaves.empty());
  bool any_uncentered = false;
  for (auto const& leaf : leaves) {
    double total = 0.0;
    for (auto v : leaf) {
      total += v;
    }
    if (std::fabs(total) > 1e-4) {
      any_uncentered = true;
    }
  }
  EXPECT_TRUE(any_uncentered)
      << "leaves came out centered without a sidecar, so the exact branch ran when it should "
         "not have";
}

/**
 * The exact builder must be correct when the matrix spans several `GHistIndexMatrix` pages.
 *
 * This is checked against GROUND TRUTH, not against the diagonal path: the root node total is
 * compared with the sum of every row's statistics, computed directly. That quantity does not
 * depend on where the quantile cuts fall, so it isolates `BuildNodeHist`'s per-page
 * accumulation from the sketcher -- which does differ between a one-page and a four-page
 * matrix, for the diagonal path too.
 *
 * The fixture asserts it really paged; a version that silently coalesced into one page would
 * prove nothing.
 */
TEST(ExactBuilder, MultiPageNodeTotalMatchesGroundTruth) {
  Context ctx;
  bst_target_t constexpr kNumClasses = 3;
  std::size_t constexpr kRows = 4096;
  bst_feature_t constexpr kCols = 8;
  auto n_free = static_cast<bst_target_t>(kNumClasses - 1);

  common::TemporaryDirectory tmpdir;
  auto fmat = RandomDataGenerator{kRows, kCols, 0.0}
                  .Seed(9)
                  .Bins(32)
                  .Batches(4)
                  .GenerateSparsePageDMatrix(tmpdir.Str() + "/cache", true);

  TrainParam param;
  param.UpdateAllowUnknown(Args{{"max_depth", "3"}, {"max_bin", "32"}, {"lambda", "1.0"}});
  int pages = 0;
  for (auto const& page : fmat->GetBatches<GHistIndexMatrix>(&ctx, HistBatch(&param))) {
    static_cast<void>(page);
    ++pages;
  }
  ASSERT_GT(pages, 1) << "the fixture did not page; this test would prove nothing";

  // Per-row statistics, derived from the row index so they do not depend on the matrix.
  GradientContainer gc;
  gc.gpair.Reshape(kRows, kNumClasses);
  gc.exact_hessian.Reshape(kRows, n_free);
  auto h_gpair = gc.gpair.HostView();
  for (std::size_t r = 0; r < kRows; ++r) {
    std::vector<double> p(kNumClasses, 1.0);
    p[r % kNumClasses] += 3.0;
    double t = 0.0;
    for (auto v : p) {
      t += v;
    }
    for (auto& v : p) {
      v /= t;
    }
    auto label = (r * 5) % kNumClasses;
    auto w = 0.5 + static_cast<double>(r % 4);
    for (bst_target_t k = 0; k < kNumClasses; ++k) {
      auto g = static_cast<float>(w * (p[k] - (label == k ? 1.0 : 0.0)));
      h_gpair(r, k) = GradientPair{g, std::max(std::fabs(g), 1e-16f)};
    }
    auto view = common::PackedHessianAtRow(gc.exact_hessian.HostRow(r), n_free, 0);
    for (std::size_t i = 0; i < n_free; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        view.Set(i, j, static_cast<float>(w * p[i] * ((i == j ? 1.0 : 0.0) - p[j])));
      }
    }
  }

  // Ground truth: sum every row exactly once.
  auto record_size = ExactHistRecordSize(n_free);
  std::vector<double> expected(record_size, 0.0);
  for (std::size_t r = 0; r < kRows; ++r) {
    for (bst_target_t i = 0; i < n_free; ++i) {
      expected[i] += static_cast<double>(h_gpair(r, i).GetGrad());
    }
    auto row = gc.exact_hessian.HostRow(r);
    for (std::size_t k = 0; k < row.size(); ++k) {
      expected[n_free + k] += static_cast<double>(row[k]);
    }
  }

  HistMakerTrainParam hist_param;
  hist_param.UpdateAllowUnknown(Args{});
  common::Monitor monitor;
  auto sampler = std::make_shared<common::ColumnSampler>();
  ExactMultiTargetHistBuilder builder{&ctx, &param, &hist_param, sampler, &monitor};
  builder.SetExactHessian(&gc.exact_hessian);

  RegTree tree{kNumClasses, kCols};
  auto gpair = gc.gpair.HostView();
  builder.InitData(fmat.get(), &tree, gpair);
  static_cast<void>(builder.InitRoot(fmat.get(), gpair, &tree));

  auto const& hist = builder.Histogram();
  ExactStatBuffer total;
  total.Reset(n_free);
  ExactNodeTotal(hist[RegTree::kRoot], n_free, hist.TotalBins(), total.Data());

  for (std::size_t i = 0; i < record_size; ++i) {
    auto scale = std::max(1.0, std::fabs(expected[i]));
    EXPECT_NEAR(total.Data()[i], expected[i], 1e-9 * scale)
        << pages << " pages, entry " << i
        << ": the node total does not equal the sum of the rows";
  }

  // Separately, drive the real updater across the same paged matrix so the full path
  // (partitioning, child histograms, subtraction, leaf materialisation) is exercised, not
  // just the root accumulation checked above.
  ObjInfo task{ObjInfo::kClassification, false, true};
  auto updater =
      std::unique_ptr<TreeUpdater>{TreeUpdater::Create("grow_quantile_histmaker", &ctx, &task)};
  updater->Configure(Args{});
  RegTree grown{kNumClasses, kCols};
  std::vector<RegTree*> trees{&grown};
  std::vector<HostDeviceVector<bst_node_t>> position(1);
  updater->Update(&param, &gc, fmat.get(),
                  common::Span<HostDeviceVector<bst_node_t>>{position.data(), position.size()},
                  trees);
  ASSERT_TRUE(grown.IsMultiTarget());
  auto leaves = LeafWeights(grown, kNumClasses);
  ASSERT_FALSE(leaves.empty()) << "no leaves were produced across " << pages << " pages";
  for (auto const& leaf : leaves) {
    double sum = 0.0;
    for (auto v : leaf) {
      EXPECT_TRUE(std::isfinite(v));
      sum += v;
    }
    EXPECT_NEAR(sum, 0.0, 1e-4) << "leaf weight is not centered";
  }
}
}  // namespace xgboost::tree
