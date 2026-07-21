/**
 * Copyright 2021-2026, XGBoost Contributors
 */
#include "../test_evaluate_splits.h"

#include <gtest/gtest.h>
#include <xgboost/base.h>        // for GradientPairPrecise, Args, Gradie...
#include <xgboost/context.h>     // for Context
#include <xgboost/data.h>        // for FeatureType, DMatrix, MetaInfo
#include <xgboost/logging.h>     // for CHECK_EQ
#include <xgboost/tree_model.h>  // for RegTree, RTreeNodeStat

#include <memory>   // for make_shared, shared_ptr, addressof
#include <numeric>  // for iota
#include <tuple>    // for make_tuple

#include "../../../../src/common/hist_util.h"           // for HistCollection, HistogramCuts
#include "../../../../src/common/random.h"              // for ColumnSampler
#include "../../../../src/common/row_set.h"             // for RowSetCollection
#include "../../../../src/data/gradient_index.h"        // for GHistIndexMatrix
#include "../../../../src/tree/hist/evaluate_splits.h"  // for HistEvaluator, TreeEvaluator
#include "../../../../src/tree/hist/expand_entry.h"     // for CPUExpandEntry
#include "../../../../src/tree/hist/hist_cache.h"       // for BoundedHistCollection
#include "../../../../src/tree/hist/hist_param.h"       // for HistMakerTrainParam
#include "../../../../src/tree/param.h"                 // for GradStats, TrainParam
#include "../../helpers.h"                              // for RandomDataGenerator, AllThreadsFo...

namespace xgboost::tree {
void TestPartitionBasedSplit::SetUp() {
  param_.UpdateAllowUnknown(Args{{"min_child_weight", "0"}, {"reg_lambda", "0"}});
  sorted_idx_.resize(n_bins_);
  std::iota(sorted_idx_.begin(), sorted_idx_.end(), 0);

  info_.num_col_ = 1;
  cuts_ = common::HistogramCuts{1};

  cuts_.SetCategorical(true, n_bins_);
  auto &h_cuts = cuts_.cut_ptrs_.HostVector();
  h_cuts[0] = 0;
  h_cuts[1] = n_bins_;
  auto &h_vals = cuts_.cut_values_.HostVector();
  h_vals.resize(n_bins_);
  std::iota(h_vals.begin(), h_vals.end(), 0.0);

  Context ctx;
  HistMakerTrainParam hist_param;
  hist_.Reset(cuts_.TotalBins(), hist_param.MaxCachedHistNodes(ctx.Device()));
  hist_.AllocateHistograms({0});
  auto node_hist = hist_[0];

  SimpleLCG lcg;
  SimpleRealUniformDistribution<double> grad_dist{-4.0, 4.0};
  SimpleRealUniformDistribution<double> hess_dist{0.0, 4.0};

  for (auto &e : node_hist) {
    e = GradientPairPrecise{grad_dist(&lcg), hess_dist(&lcg)};
    total_gpair_ += e;
  }

  auto enumerate = [this, n_feat = info_.num_col_](common::GHistRow hist,
                                                   GradientPairPrecise parent_sum) {
    int32_t best_thresh = -1;
    float best_score{-std::numeric_limits<float>::infinity()};
    TreeEvaluator evaluator{param_, static_cast<bst_feature_t>(n_feat), DeviceOrd::CPU(), 1u};
    auto tree_evaluator = evaluator.GetEvaluator();
    GradientPairPrecise left_sum;
    auto parent_gain = tree_evaluator.CalcGain(0, param_, GradStats{total_gpair_});
    for (size_t i = 0; i < hist.size() - 1; ++i) {
      left_sum += hist[i];
      auto right_sum = parent_sum - left_sum;
      auto gain =
          tree_evaluator.CalcSplitGain(param_, 0, 0, GradStats{left_sum}, GradStats{right_sum}) -
          parent_gain;
      if (gain > best_score) {
        best_score = gain;
        best_thresh = i;
      }
    }
    return std::make_tuple(best_thresh, best_score);
  };

  // enumerate all possible partitions to find the optimal split
  do {
    std::vector<GradientPairPrecise> sorted_hist(node_hist.size());
    for (size_t i = 0; i < sorted_hist.size(); ++i) {
      sorted_hist[i] = node_hist[sorted_idx_[i]];
    }
    auto [thresh, score] = enumerate({sorted_hist}, total_gpair_);
    if (score > best_score_) {
      best_score_ = score;
    }
  } while (std::next_permutation(sorted_idx_.begin(), sorted_idx_.end()));
}

void TestEvaluateSplits(bool force_read_by_column) {
  Context ctx;
  ctx.nthread = 4;
  static constexpr bst_idx_t kRows = 8, kCols = 16;
  auto sampler = std::make_shared<common::ColumnSampler>();

  TrainParam param;
  param.UpdateAllowUnknown(Args{{"min_child_weight", "0"}, {"reg_lambda", "0"}});

  auto dmat = RandomDataGenerator(kRows, kCols, 0).Seed(3).GenerateDMatrix();

  auto evaluator = HistEvaluator{&ctx, &param, dmat->Info(), sampler};
  BoundedHistCollection hist;
  std::vector<GradientPair> row_gpairs = {{1.23f, 0.24f},  {0.24f, 0.25f}, {0.26f, 0.27f},
                                          {2.27f, 0.28f},  {0.27f, 0.29f}, {0.37f, 0.39f},
                                          {-0.47f, 0.49f}, {0.57f, 0.59f}};

  size_t constexpr kMaxBins = 4;
  // dense, no missing values
  GHistIndexMatrix gmat(&ctx, dmat.get(), kMaxBins, 0.5, false);
  common::RowSetCollection row_set_collection;
  std::vector<bst_idx_t> &row_indices = *row_set_collection.Data();
  row_indices.resize(kRows);
  std::iota(row_indices.begin(), row_indices.end(), 0);
  row_set_collection.Init();

  HistMakerTrainParam hist_param;
  hist.Reset(gmat.cut.Ptrs().back(), hist_param.MaxCachedHistNodes(ctx.Device()));
  hist.AllocateHistograms({0});
  auto const &elem = row_set_collection[0];
  common::BuildHist<false>(row_gpairs, common::Span{elem.begin(), elem.end()}, gmat, hist[0],
                           force_read_by_column);

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

  auto best_loss_chg = evaluator.Evaluator().CalcSplitGain(
                           param, 0, entries.front().split.SplitIndex(),
                           entries.front().split.left_sum, entries.front().split.right_sum) -
                       evaluator.Stats().front().root_gain;
  ASSERT_EQ(entries.front().split.loss_chg, best_loss_chg);
  ASSERT_GT(entries.front().split.loss_chg, 16.2f);

  // Assert that's the best split
  for (size_t i = 1; i < gmat.cut.Ptrs().size(); ++i) {
    GradStats left, right;
    for (size_t j = gmat.cut.Ptrs()[i - 1]; j < gmat.cut.Ptrs()[i]; ++j) {
      auto loss_chg = evaluator.Evaluator().CalcSplitGain(param, 0, i - 1, left, right) -
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

  HistMultiEvaluator evaluator{&ctx, p_fmat->Info(), &param, n_targets, sampler};
  HistMakerTrainParam hist_param;
  std::vector<BoundedHistCollection> histogram(n_targets);
  linalg::Vector<GradientPairPrecise> root_sum({2}, DeviceOrd::CPU());
  for (bst_target_t t{0}; t < n_targets; ++t) {
    auto &hist = histogram[t];
    hist.Reset(n_bins * n_features, hist_param.MaxCachedHistNodes(ctx.Device()));
    hist.AllocateHistograms({0});
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
  // Compute root sum_hess by summing hessians across all targets
  float root_sum_hess = 0.0f;
  for (bst_target_t t{0}; t < n_targets; ++t) {
    root_sum_hess += static_cast<float>(root_sum.HostView()(t).GetHess());
  }
  tree.SetRoot(weight.HostView(), root_sum_hess);
  auto w = weight.HostView();
  ASSERT_EQ(w.Size(), n_targets);
  ASSERT_EQ(w(0), -1.5);
  ASSERT_EQ(w(1), -1.5);

  common::HistogramCuts cuts{2};
  cuts.cut_ptrs_ = {0, 2, 4};
  cuts.cut_values_ = {0.5, 1.0, 2.0, 3.0};

  std::vector<MultiExpandEntry> entries(1, {/*nidx=*/0, /*depth=*/0});

  std::vector<BoundedHistCollection const *> ptrs;
  std::transform(histogram.cbegin(), histogram.cend(), std::back_inserter(ptrs),
                 [](auto const &h) { return std::addressof(h); });

  evaluator.EvaluateSplits(tree, ptrs, cuts, {}, &entries);

  ASSERT_EQ(entries.front().split.loss_chg, 12.5);
  ASSERT_EQ(entries.front().split.split_value, 0.5);
  ASSERT_EQ(entries.front().split.SplitIndex(), 0);

  ASSERT_EQ(sampler->GetFeatureSet(&ctx, 0)->Size(), n_features);
}

TEST(HistEvaluator, Apply) {
  Context ctx;
  ctx.nthread = 4;
  RegTree tree;
  static constexpr bst_idx_t kRows = 8, kCols = 16;
  TrainParam param;
  param.UpdateAllowUnknown(Args{{"min_child_weight", "0"}, {"reg_lambda", "0.0"}});
  auto dmat = RandomDataGenerator(kRows, kCols, 0).Seed(3).GenerateDMatrix();
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
  static constexpr bst_idx_t kRows = 128, kCols = 1;
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
  HistMakerTrainParam hist_param;

  for (auto const &gmat : dmat->GetBatches<GHistIndexMatrix>(&ctx, {32, param.sparse_threshold})) {
    BoundedHistCollection hist;

    entries.front().nid = 0;
    entries.front().depth = 0;

    hist.Reset(gmat.cut.TotalBins(), hist_param.MaxCachedHistNodes(ctx.Device()));
    hist.AllocateHistograms({0});
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
  Context ctx;
  BoundedHistCollection hist;
  HistMakerTrainParam hist_param;
  hist.Reset(cuts_.TotalBins(), hist_param.MaxCachedHistNodes(ctx.Device()));
  hist.AllocateHistograms({0});
  auto node_hist = hist[0];
  ASSERT_EQ(node_hist.size(), feature_histogram_.size());
  std::copy(feature_histogram_.cbegin(), feature_histogram_.cend(), node_hist.begin());

  auto sampler = std::make_shared<common::ColumnSampler>();
  MetaInfo info;
  info.num_col_ = 1;
  info.feature_types = {FeatureType::kCategorical};

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

namespace {
class TestHistMultiEvaluator : public ::testing::Test {
 protected:
  static constexpr bst_feature_t kNFeatures{1};
  static constexpr bst_target_t kNTargets{2};
  static constexpr bst_bin_t kNCats{3};

  Context ctx_;
  TrainParam param_;
  std::shared_ptr<common::ColumnSampler> sampler_{std::make_shared<common::ColumnSampler>()};
  MetaInfo info_;
  std::vector<BoundedHistCollection> histogram_ = std::vector<BoundedHistCollection>(kNTargets);
  linalg::Vector<GradientPairPrecise> root_sum_{{kNTargets}, DeviceOrd::CPU()};
  common::HistogramCuts cuts_{kNFeatures};
  RegTree tree_{kNTargets, kNFeatures};
  std::vector<MultiExpandEntry> entries_ = std::vector<MultiExpandEntry>(1, {0, 0});
  std::vector<BoundedHistCollection const *> histogram_ptrs_;
  std::unique_ptr<HistMultiEvaluator> evaluator_;

  void SetUp() override {
    ctx_.nthread = 1;
    info_.num_col_ = kNFeatures;
    info_.feature_types = {FeatureType::kCategorical};

    cuts_.cut_ptrs_ = {0, kNCats};
    cuts_.cut_values_ = {0.0, 1.0, 2.0};
    cuts_.SetCategorical(true, 2.0);

    HistMakerTrainParam hist_param;
    for (auto &hist : histogram_) {
      hist.Reset(cuts_.TotalBins(), hist_param.MaxCachedHistNodes(ctx_.Device()));
      hist.AllocateHistograms({0});
      histogram_ptrs_.push_back(&hist);
    }
  }

  void SetHistData(std::vector<std::vector<GradientPairPrecise>> const &hist_data) {
    ASSERT_EQ(hist_data.size(), kNTargets);
    for (bst_target_t t = 0; t < kNTargets; ++t) {
      ASSERT_EQ(hist_data[t].size(), kNCats);
      root_sum_(t) = GradientPairPrecise{};
      auto node_hist = histogram_[t][0];
      for (bst_bin_t b = 0; b < kNCats; ++b) {
        node_hist[b] = hist_data[t][b];
        root_sum_(t) += node_hist[b];
      }
    }
  }

  void SetOneHotData() {
    this->SetHistData({{{1.0, 0.5}, {-0.5, 0.5}, {0.5, 0.5}},    // t-0
                       {{0.5, 0.5}, {1.0, 0.5}, {-0.5, 0.5}}});  // t-1
  }

  void EvaluateSplits(Args args = {}) {
    param_.UpdateAllowUnknown(Args{{"min_child_weight", "0"},
                                   {"reg_alpha", "0"},
                                   {"reg_lambda", "0"},
                                   {"max_delta_step", "0"},
                                   {"max_cat_to_onehot", "4"}});
    param_.UpdateAllowUnknown(args);
    evaluator_ = std::make_unique<HistMultiEvaluator>(&ctx_, info_, &param_, kNTargets, sampler_);
    entries_.clear();
    entries_.emplace_back(0, 0);

    auto weight = evaluator_->InitRoot(root_sum_.HostView());
    float root_sum_hess = 0.0f;
    for (bst_target_t t = 0; t < kNTargets; ++t) {
      root_sum_hess += static_cast<float>(root_sum_(t).GetHess());
    }
    tree_.SetRoot(weight.HostView(), root_sum_hess);

    evaluator_->EvaluateSplits(tree_, histogram_ptrs_, cuts_, info_.feature_types.ConstHostSpan(),
                               &entries_);
  }

  void ApplyTreeSplit() {
    ASSERT_TRUE(entries_.front().split.is_cat);
    evaluator_->ApplyTreeSplit(entries_.front(), &tree_);
    ASSERT_TRUE(tree_.HasCategoricalSplit());
    auto mt_view = tree_.HostMtView();
    ASSERT_EQ(mt_view.SplitType(0), FeatureType::kCategorical);
    ASSERT_FALSE(mt_view.NodeCats(0).empty());
  }
};
}  // anonymous namespace

TEST_F(TestHistMultiEvaluator, CategoricalOneHot) {
  this->SetOneHotData();
  this->EvaluateSplits(Args{{"reg_alpha", "0.1"}, {"reg_lambda", "1"}, {"max_delta_step", "0.25"}});

  auto const &split = entries_.front().split;
  ASSERT_TRUE(split.is_cat);
  ASSERT_FALSE(split.cat_bits.empty());
  ASSERT_EQ(split.split_value, 1.0f);
  ASSERT_NEAR(split.loss_chg, 0.45f, 1e-6f);

  common::KCatBitField cat_bits{split.cat_bits};
  auto chosen_cat = static_cast<bst_cat_t>(split.split_value);
  ASSERT_TRUE(cat_bits.Check(chosen_cat));

  // Verify ApplyTreeSplit works with categorical split.
  this->ApplyTreeSplit();
}

TEST_F(TestHistMultiEvaluator, CategoricalPartition) {
  this->SetHistData({{{-3.0, 1.0}, {-3.0, 1.0}, {-3.0, 1.0}},    // t-0
                     {{-3.0, 1.0}, {-3.0, 1.0}, {-2.0, 1.0}}});  // t-1
  // Include gradients for missing feature values. The optimal split is found by the
  // backward scan, with missing values assigned to the right child.
  root_sum_(0) += GradientPairPrecise{-3.0, 1.0};
  root_sum_(1) += GradientPairPrecise{-2.0, 1.0};
  this->EvaluateSplits(Args{{"max_cat_to_onehot", "1"}});

  auto const &split = entries_.front().split;
  ASSERT_TRUE(split.is_cat);
  ASSERT_FLOAT_EQ(split.loss_chg, 1.0f);
  ASSERT_FALSE(split.DefaultLeft());
  ASSERT_EQ(split.left_sum.size(), kNTargets);
  ASSERT_EQ(split.right_sum.size(), kNTargets);

  common::KCatBitField cat_bits{split.cat_bits};
  ASSERT_FALSE(cat_bits.Check(0));
  ASSERT_FALSE(cat_bits.Check(1));
  ASSERT_TRUE(cat_bits.Check(2));
  ASSERT_EQ(split.right_sum[0], GradientPairPrecise(-6.0, 2.0));
  ASSERT_EQ(split.right_sum[1], GradientPairPrecise(-4.0, 2.0));
  for (bst_target_t t = 0; t < kNTargets; ++t) {
    ASSERT_EQ(split.left_sum[t] + split.right_sum[t], root_sum_(t));
  }

  this->ApplyTreeSplit();
}

}  // namespace xgboost::tree
