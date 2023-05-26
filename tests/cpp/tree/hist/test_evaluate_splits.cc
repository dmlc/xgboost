/**
 * Copyright 2021-2023 by XGBoost Contributors
 */
#include "../test_evaluate_splits.h"

#include <gtest/gtest.h>
#include <xgboost/base.h>                               // for GradientPairPrecise, Args, Gradie...
#include <xgboost/context.h>                            // for Context
#include <xgboost/data.h>                               // for FeatureType, DMatrix, MetaInfo
#include <xgboost/logging.h>                            // for CHECK_EQ
#include <xgboost/tree_model.h>                         // for RegTree, RTreeNodeStat

#include <memory>                                       // for make_shared, shared_ptr, addressof

#include "../../../../src/common/hist_util.h"           // for HistCollection, HistogramCuts
#include "../../../../src/common/random.h"              // for ColumnSampler
#include "../../../../src/common/row_set.h"             // for RowSetCollection
#include "../../../../src/data/gradient_index.h"        // for GHistIndexMatrix
#include "../../../../src/tree/hist/evaluate_splits.h"  // for HistEvaluator
#include "../../../../src/tree/hist/expand_entry.h"     // for CPUExpandEntry
#include "../../../../src/tree/param.h"                 // for GradStats, TrainParam
#include "../../helpers.h"                              // for RandomDataGenerator, AllThreadsFo...

namespace xgboost::tree {
void TestEvaluateSplits(bool force_read_by_column) {
  Context ctx;
  ctx.nthread = 4;
  int static constexpr kRows = 8, kCols = 16;
  auto sampler = std::make_shared<common::ColumnSampler>();

  TrainParam param;
  param.UpdateAllowUnknown(Args{{"min_child_weight", "0"}, {"reg_lambda", "0"}});

  auto dmat = RandomDataGenerator(kRows, kCols, 0).Seed(3).GenerateDMatrix();

  auto evaluator = HistEvaluator{&ctx, &param, dmat->Info(), sampler};
  common::HistCollection hist;
  std::vector<GradientPair> row_gpairs = {
      {1.23f, 0.24f}, {0.24f, 0.25f}, {0.26f, 0.27f},  {2.27f, 0.28f},
      {0.27f, 0.29f}, {0.37f, 0.39f}, {-0.47f, 0.49f}, {0.57f, 0.59f}};

  size_t constexpr kMaxBins = 4;
  // dense, no missing values
  GHistIndexMatrix gmat(&ctx, dmat.get(), kMaxBins, 0.5, false);
  common::RowSetCollection row_set_collection;
  std::vector<size_t> &row_indices = *row_set_collection.Data();
  row_indices.resize(kRows);
  std::iota(row_indices.begin(), row_indices.end(), 0);
  row_set_collection.Init();

  auto hist_builder = common::GHistBuilder(gmat.cut.Ptrs().back());
  hist.Init(gmat.cut.Ptrs().back());
  hist.AddHistRow(0);
  hist.AllocateAllData();
  hist_builder.template BuildHist<false>(row_gpairs, row_set_collection[0],
                                         gmat, hist[0], force_read_by_column);

  // Compute total gradient for all data points
  GradientPairPrecise total_gpair;
  for (const auto &e : row_gpairs) {
    total_gpair += GradientPairPrecise(e);
  }

  RegTree tree;
  std::vector<CPUExpandEntry> entries(1);
  entries.front().nid = 0;
  entries.front().depth = 0;

  evaluator.InitRoot(GradStats{total_gpair});
  evaluator.EvaluateSplits(hist, gmat.cut, {}, tree, &entries);

  auto best_loss_chg =
      evaluator.Evaluator().CalcSplitGain(
          param, 0, entries.front().split.SplitIndex(),
          entries.front().split.left_sum, entries.front().split.right_sum) -
      evaluator.Stats().front().root_gain;
  ASSERT_EQ(entries.front().split.loss_chg, best_loss_chg);
  ASSERT_GT(entries.front().split.loss_chg, 16.2f);

  // Assert that's the best split
  for (size_t i = 1; i < gmat.cut.Ptrs().size(); ++i) {
    GradStats left, right;
    for (size_t j = gmat.cut.Ptrs()[i-1]; j < gmat.cut.Ptrs()[i]; ++j) {
      auto loss_chg =
          evaluator.Evaluator().CalcSplitGain(param, 0, i - 1, left, right) -
          evaluator.Stats().front().root_gain;
      ASSERT_GE(best_loss_chg, loss_chg);
      left.Add(hist[0][j].GetGrad(), hist[0][j].GetHess());
      right.SetSubstract(GradStats{total_gpair}, left);
    }
  }
}

TEST(HistEvaluator, Evaluate) {
  TestEvaluateSplits(false);
  TestEvaluateSplits(true);
}

TEST(HistMultiEvaluator, Evaluate) {
  Context ctx;
  ctx.nthread = 1;

  TrainParam param;
  param.Init(Args{{"min_child_weight", "0"}, {"reg_lambda", "0"}});
  auto sampler = std::make_shared<common::ColumnSampler>();

  std::size_t n_samples = 3;
  bst_feature_t n_features = 2;
  bst_target_t n_targets = 2;
  bst_bin_t n_bins = 2;

  auto p_fmat =
      RandomDataGenerator{n_samples, n_features, 0.5}.Targets(n_targets).GenerateDMatrix(true);

  HistMultiEvaluator evaluator{&ctx, p_fmat->Info(), &param, sampler};
  std::vector<common::HistCollection> histogram(n_targets);
  linalg::Vector<GradientPairPrecise> root_sum({2}, Context::kCpuId);
  for (bst_target_t t{0}; t < n_targets; ++t) {
    auto &hist = histogram[t];
    hist.Init(n_bins * n_features);
    hist.AddHistRow(0);
    hist.AllocateAllData();
    auto node_hist = hist[0];
    node_hist[0] = {-0.5, 0.5};
    node_hist[1] = {2.0, 0.5};
    node_hist[2] = {0.5, 0.5};
    node_hist[3] = {1.0, 0.5};

    root_sum(t) += node_hist[0];
    root_sum(t) += node_hist[1];
  }

  RegTree tree{n_targets, n_features};
  auto weight = evaluator.InitRoot(root_sum.HostView());
  tree.SetLeaf(RegTree::kRoot, weight.HostView());
  auto w = weight.HostView();
  ASSERT_EQ(w.Size(), n_targets);
  ASSERT_EQ(w(0), -1.5);
  ASSERT_EQ(w(1), -1.5);

  common::HistogramCuts cuts;
  cuts.cut_ptrs_ = {0, 2, 4};
  cuts.cut_values_ = {0.5, 1.0, 2.0, 3.0};
  cuts.min_vals_ = {-0.2, 1.8};

  std::vector<MultiExpandEntry> entries(1, {/*nidx=*/0, /*depth=*/0});

  std::vector<common::HistCollection const *> ptrs;
  std::transform(histogram.cbegin(), histogram.cend(), std::back_inserter(ptrs),
                 [](auto const &h) { return std::addressof(h); });

  evaluator.EvaluateSplits(tree, ptrs, cuts, &entries);

  ASSERT_EQ(entries.front().split.loss_chg, 12.5);
  ASSERT_EQ(entries.front().split.split_value, 0.5);
  ASSERT_EQ(entries.front().split.SplitIndex(), 0);

  ASSERT_EQ(sampler->GetFeatureSet(0)->Size(), n_features);
}

TEST(HistEvaluator, Apply) {
  Context ctx;
  ctx.nthread = 4;
  RegTree tree;
  int static constexpr kNRows = 8, kNCols = 16;
  TrainParam param;
  param.UpdateAllowUnknown(Args{{"min_child_weight", "0"}, {"reg_lambda", "0.0"}});
  auto dmat = RandomDataGenerator(kNRows, kNCols, 0).Seed(3).GenerateDMatrix();
  auto sampler = std::make_shared<common::ColumnSampler>();
  auto evaluator_ = HistEvaluator{&ctx, &param, dmat->Info(), sampler};

  CPUExpandEntry entry{0, 0};
  entry.split.loss_chg = 10.0f;
  entry.split.left_sum = GradStats{0.4, 0.6f};
  entry.split.right_sum = GradStats{0.5, 0.5f};

  evaluator_.ApplyTreeSplit(entry, &tree);
  ASSERT_EQ(tree.NumExtraNodes(), 2);
  ASSERT_EQ(tree.Stat(tree[0].LeftChild()).sum_hess, 0.6f);
  ASSERT_EQ(tree.Stat(tree[0].RightChild()).sum_hess, 0.5f);

  {
    RegTree tree;
    entry.split.is_cat = true;
    entry.split.split_value = 1.0;
    evaluator_.ApplyTreeSplit(entry, &tree);
    auto l = entry.split.left_sum;
    ASSERT_NEAR(tree[1].LeafValue(), -l.sum_grad / l.sum_hess * param.learning_rate, kRtEps);
    ASSERT_NEAR(tree[2].LeafValue(), -param.learning_rate, kRtEps);
  }
}

TEST_F(TestPartitionBasedSplit, CPUHist) {
  Context ctx;
  // check the evaluator is returning the optimal split
  std::vector<FeatureType> ft{FeatureType::kCategorical};
  auto sampler = std::make_shared<common::ColumnSampler>();
  HistEvaluator evaluator{&ctx, &param_, info_, sampler};
  evaluator.InitRoot(GradStats{total_gpair_});
  RegTree tree;
  std::vector<CPUExpandEntry> entries(1);
  evaluator.EvaluateSplits(hist_, cuts_, {ft}, tree, &entries);
  ASSERT_NEAR(entries[0].split.loss_chg, best_score_, 1e-16);
}

namespace {
auto CompareOneHotAndPartition(bool onehot) {
  Context ctx;
  int static constexpr kRows = 128, kCols = 1;
  std::vector<FeatureType> ft(kCols, FeatureType::kCategorical);

  TrainParam param;
  if (onehot) {
    // force use one-hot
    param.UpdateAllowUnknown(
        Args{{"min_child_weight", "0"}, {"reg_lambda", "0"}, {"max_cat_to_onehot", "100"}});
  } else {
    param.UpdateAllowUnknown(
        Args{{"min_child_weight", "0"}, {"reg_lambda", "0"}, {"max_cat_to_onehot", "1"}});
  }

  size_t n_cats{2};

  auto dmat =
      RandomDataGenerator(kRows, kCols, 0).Seed(3).Type(ft).MaxCategory(n_cats).GenerateDMatrix();

  auto sampler = std::make_shared<common::ColumnSampler>();
  auto evaluator = HistEvaluator{&ctx, &param, dmat->Info(), sampler};
  std::vector<CPUExpandEntry> entries(1);

  for (auto const &gmat : dmat->GetBatches<GHistIndexMatrix>(&ctx, {32, param.sparse_threshold})) {
    common::HistCollection hist;

    entries.front().nid = 0;
    entries.front().depth = 0;

    hist.Init(gmat.cut.TotalBins());
    hist.AddHistRow(0);
    hist.AllocateAllData();
    auto node_hist = hist[0];

    CHECK_EQ(node_hist.size(), n_cats);
    CHECK_EQ(node_hist.size(), gmat.cut.Ptrs().back());

    GradientPairPrecise total_gpair;
    for (size_t i = 0; i < node_hist.size(); ++i) {
      node_hist[i] = {static_cast<double>(node_hist.size() - i), 1.0};
      total_gpair += node_hist[i];
    }
    RegTree tree;
    evaluator.InitRoot(GradStats{total_gpair});
    evaluator.EvaluateSplits(hist, gmat.cut, ft, tree, &entries);
  }
  return entries.front();
}
}  // anonymous namespace

TEST(HistEvaluator, Categorical) {
  auto with_onehot = CompareOneHotAndPartition(true);
  auto with_part = CompareOneHotAndPartition(false);

  ASSERT_EQ(with_onehot.split.loss_chg, with_part.split.loss_chg);
}

TEST_F(TestCategoricalSplitWithMissing, HistEvaluator) {
  common::HistCollection hist;
  hist.Init(cuts_.TotalBins());
  hist.AddHistRow(0);
  hist.AllocateAllData();
  auto node_hist = hist[0];
  ASSERT_EQ(node_hist.size(), feature_histogram_.size());
  std::copy(feature_histogram_.cbegin(), feature_histogram_.cend(), node_hist.begin());

  auto sampler = std::make_shared<common::ColumnSampler>();
  MetaInfo info;
  info.num_col_ = 1;
  info.feature_types = {FeatureType::kCategorical};
  Context ctx;
  auto evaluator = HistEvaluator{&ctx, &param_, info, sampler};
  evaluator.InitRoot(GradStats{parent_sum_});

  std::vector<CPUExpandEntry> entries(1);
  RegTree tree;
  evaluator.EvaluateSplits(hist, cuts_, info.feature_types.ConstHostSpan(), tree, &entries);
  auto const &split = entries.front().split;

  this->CheckResult(split.loss_chg, split.SplitIndex(), split.split_value, split.is_cat,
                    split.DefaultLeft(),
                    GradientPairPrecise{split.left_sum.GetGrad(), split.left_sum.GetHess()},
                    GradientPairPrecise{split.right_sum.GetGrad(), split.right_sum.GetHess()});
}
}  // namespace xgboost::tree
