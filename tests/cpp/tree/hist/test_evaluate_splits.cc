/**
 * Copyright 2021-2026, XGBoost Contributors
 */
#include "../test_evaluate_splits.h"

#include <gtest/gtest.h>
#include <xgboost/base.h>        // for Args, GradientPairPrecise
#include <xgboost/context.h>     // for Context
#include <xgboost/data.h>        // for FeatureType, MetaInfo
#include <xgboost/tree_model.h>  // for RegTree

#include <initializer_list>  // for initializer_list
#include <memory>            // for make_shared, make_unique
#include <numeric>           // for accumulate, iota
#include <tuple>             // for make_tuple
#include <type_traits>       // for is_same_v

#include "../../../../src/common/random.h"              // for ColumnSampler
#include "../../../../src/tree/hist/evaluate_splits.h"  // for HistEvaluatorImpl
#include "../../../../src/tree/hist/hist_param.h"       // for HistMakerTrainParam
#include "../../../../src/tree/tree_view.h"             // for ScalarTreeView
#include "../../helpers.h"  // for SimpleLCG, SimpleRealUniformDistribution

namespace xgboost::tree {
void TestPartitionBasedSplit::SetUp() {
  param_.UpdateAllowUnknown(
      Args{{"min_child_weight", "0"}, {"reg_lambda", "0"}, {"max_cat_to_onehot", "1"}});
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

TEST_F(TestPartitionBasedSplit, CPUHist) {
  Context ctx;
  // check the evaluator is returning the optimal split
  std::vector<FeatureType> ft{FeatureType::kCategorical};
  auto sampler = std::make_shared<common::ColumnSampler>();
  HistEvaluator evaluator{&ctx, &param_, info_, sampler};
  evaluator.InitRoot(GradStats{total_gpair_});
  std::vector<CPUExpandEntry> entries(1);
  evaluator.EvaluateSplits(hist_, cuts_, {ft}, &entries);
  ASSERT_FLOAT_EQ(entries[0].split.loss_chg, best_score_);
}

namespace {
template <typename EntryT, bst_target_t split_targets, bst_target_t targets = split_targets>
struct EvaluatorTestCase {
  using Entry = EntryT;
  static constexpr bst_target_t kSplitTargets = split_targets;
  static constexpr bst_target_t kTargets = targets;
};

template <typename TestCase>
class TestHistEvaluator : public ::testing::Test {
 protected:
  using Entry = typename TestCase::Entry;
  static constexpr auto kSplitTargets = TestCase::kSplitTargets;
  static constexpr auto kTargets = TestCase::kTargets;
  static constexpr bool kScalar = std::is_same_v<Entry, CPUExpandEntry>;
  Context ctx_;
  TrainParam param_;
  common::HistogramCuts cuts_ = MakeCutsForTest({0.0, 1.0, 2.0}, {0, 3}, DeviceOrd::CPU());
  std::vector<FeatureType> ft_{FeatureType::kCategorical};
  std::unique_ptr<HistEvaluatorImpl<Entry>> evaluator_;
  std::vector<Entry> entries_;
  linalg::Vector<float> weight_{{kSplitTargets}, DeviceOrd::CPU()};
  float root_hess_{0.0f};

  void SetUp() override { ctx_.nthread = 1; }

  // A single histogram/missing sum is repeated across targets; otherwise use each target's data.
  auto const &Evaluate(std::vector<std::vector<GradientPairPrecise>> const &hist_data,
                       std::vector<GradientPairPrecise> const &missing = {}, Args args = {}) {
    param_.UpdateAllowUnknown(Args{{"min_child_weight", "0"},
                                   {"reg_lambda", "0"},
                                   {"reg_alpha", "0"},
                                   {"max_delta_step", "0"},
                                   {"max_cat_to_onehot", "1"},
                                   {"max_cat_threshold", "64"}});
    param_.UpdateAllowUnknown(args);
    MetaInfo info;
    info.num_col_ = ft_.size();
    auto sampler = std::make_shared<common::ColumnSampler>();
    if constexpr (kScalar) {
      evaluator_ = std::make_unique<HistEvaluatorImpl<Entry>>(&ctx_, &param_, info, sampler);
    } else {
      evaluator_ =
          std::make_unique<HistEvaluatorImpl<Entry>>(&ctx_, info, &param_, kSplitTargets, sampler);
    }
    std::vector<BoundedHistCollection> hist(kSplitTargets);
    std::vector<BoundedHistCollection const *> ptrs;
    linalg::Vector<GradientPairPrecise> parent({kSplitTargets}, ctx_.Device());
    root_hess_ = 0.0f;
    for (bst_target_t t = 0; t < kSplitTargets; ++t) {
      auto const &bins = hist_data.at(hist_data.size() == 1 ? 0 : t);
      CHECK_EQ(bins.size(), cuts_.TotalBins());
      hist[t].Reset(cuts_.TotalBins(), 2);
      hist[t].AllocateHistograms({0});
      std::copy(bins.begin(), bins.end(), hist[t][0].begin());
      ptrs.push_back(&hist[t]);
      auto sum = missing.empty() ? GradientPairPrecise{} : missing.at(missing.size() == 1 ? 0 : t);
      parent(t) = std::accumulate(bins.begin(), bins.begin() + cuts_.Ptrs()[1], sum);
      root_hess_ += parent(t).GetHess();
    }
    entries_.assign(1, Entry{0, 0});
    if constexpr (kScalar) {
      weight_(0) = evaluator_->InitRoot(GradStats{parent(0)});
      EXPECT_FLOAT_EQ(weight_(0), CalcWeight(param_, parent(0)));
      evaluator_->EvaluateSplits(hist[0], cuts_, ft_, &entries_);
    } else {
      weight_ = evaluator_->InitRoot(parent.HostView());
      for (bst_target_t t = 0; t < kSplitTargets; ++t) {
        EXPECT_FLOAT_EQ(weight_(t), CalcWeight(param_, parent(t)));
      }
      evaluator_->EvaluateSplits(ptrs, cuts_, ft_, &entries_);
    }
    EXPECT_EQ(sampler->GetFeatureSet(&ctx_, 0)->Size(), ft_.size());
    return entries_.front().split;
  }

  static auto Stats(decltype(Entry{}.split.left_sum) const &sum, bst_target_t t) {
    if constexpr (kScalar) {
      return GradientPairPrecise{sum.GetGrad(), sum.GetHess()};
    } else {
      return sum.at(t);
    }
  }

  void CheckCategories(std::initializer_list<bool> expected) {
    auto const &split = entries_.front().split;
    ASSERT_TRUE(split.is_cat);
    common::KCatBitField bits{split.cat_bits};
    bst_cat_t cat = 0;
    for (bool selected : expected) {
      EXPECT_EQ(bits.Check(cat++), selected);
    }
  }

  void CheckApply() {
    RegTree tree{kTargets, static_cast<bst_feature_t>(ft_.size())};
    if constexpr (kScalar) {
      tree.SetRoot(weight_(0), root_hess_);
    } else {
      tree.SetRoot(weight_.HostView(), root_hess_);
    }
    evaluator_->ApplyTreeSplit(entries_.front(), &tree);
    EXPECT_EQ(tree.NumTargets(), kTargets);
    if constexpr (!kScalar) {
      EXPECT_EQ(tree.GetMultiTargetTree()->NumSplitTargets(), kSplitTargets);
    }
    // Projected leaves are finalized by the updater using full gradients.
    if constexpr (kSplitTargets == kTargets) {
      tree.FinalizeLeaves(param_.learning_rate);
    }
    auto view = [&] {
      if constexpr (kScalar) {
        return ScalarTreeView{&tree};
      } else {
        return tree.HostMtView();
      }
    }();
    auto const &split = entries_.front().split;
    ASSERT_EQ(view.Size(), 3);
    EXPECT_EQ(view.SplitIndex(0), split.SplitIndex());
    EXPECT_EQ(view.DefaultLeft(0), split.DefaultLeft());
    EXPECT_EQ(view.SplitType(0),
              split.is_cat ? FeatureType::kCategorical : FeatureType::kNumerical);
    EXPECT_EQ(view.NodeCats(0).size(), split.cat_bits.size());
    for (auto child : {view.LeftChild(0), view.RightChild(0)}) {
      auto const &sum = child == view.LeftChild(0) ? split.left_sum : split.right_sum;
      double hess = 0.0;
      for (bst_target_t t = 0; t < kSplitTargets; ++t) {
        auto stats = Stats(sum, t);
        hess += stats.GetHess();
        if constexpr (kSplitTargets == kTargets) {
          auto weight = [&] {
            if constexpr (kScalar) {
              return view.LeafValue(child);
            } else {
              return view.LeafValue(child)(t);
            }
          }();
          EXPECT_NEAR(weight, CalcWeight(param_, stats) * param_.learning_rate, kRtEps);
        }
      }
      EXPECT_FLOAT_EQ(view.SumHess(child), hess);
    }
  }
};

using HistEvaluatorCases = ::testing::Types<
    EvaluatorTestCase<CPUExpandEntry, 1>,
    EvaluatorTestCase<MultiExpandEntry, 1, 2>,  // one projected gradient, two outputs
    EvaluatorTestCase<MultiExpandEntry, 2>>;
TYPED_TEST_SUITE(TestHistEvaluator, HistEvaluatorCases);

TYPED_TEST(TestHistEvaluator, Numeric) {
  for (auto threads : {1, 4}) {
    this->ctx_.nthread = threads;
    this->ft_ = {FeatureType::kNumerical, FeatureType::kNumerical};
    this->cuts_ = MakeCutsForTest({0.5, 1.0, 2.0, 3.0}, {0, 2, 4}, DeviceOrd::CPU());
    auto const &split = this->Evaluate({{{-0.5, 0.5}, {2.0, 0.5}, {0.5, 0.5}, {1.0, 0.5}}});
    EXPECT_FLOAT_EQ(split.loss_chg, 6.25f * this->kSplitTargets);
    EXPECT_EQ(split.SplitIndex(), 0);
    EXPECT_EQ(split.split_value, 0.5f);
    this->CheckApply();

    this->ft_.resize(1);
    this->cuts_ = MakeCutsForTest({0.0, 1.0, 2.0}, {0, 3}, DeviceOrd::CPU());
    for (auto missing : {-3.0, 3.0}) {
      auto const &split = this->Evaluate({{{-2.0, 1.0}, {0.0, 1.0}, {2.0, 1.0}}}, {{missing, 1.0}});
      EXPECT_FLOAT_EQ(split.loss_chg, 12.25f * this->kSplitTargets);
      EXPECT_EQ(split.DefaultLeft(), missing < 0.0);
      EXPECT_FALSE(split.is_cat);
      EXPECT_EQ(split.split_value, missing < 0.0 ? 0.0f : 1.0f);
      for (bst_target_t t = 0; t < this->kSplitTargets; ++t) {
        EXPECT_EQ(this->Stats(split.left_sum, t),
                  GradientPairPrecise(missing < 0.0 ? -5.0 : -2.0, 2.0));
        EXPECT_EQ(this->Stats(split.right_sum, t),
                  GradientPairPrecise(missing < 0.0 ? 2.0 : 5.0, 2.0));
      }
    }
  }
}

TYPED_TEST(TestHistEvaluator, OneHot) {
  for (auto missing : {-3.0, 3.0}) {
    auto const &split = this->Evaluate({{{1.0, 1.0}, {-2.0, 1.0}, {1.0, 1.0}}}, {{missing, 1.0}},
                                       {{"max_cat_to_onehot", "4"}});
    // Isolate category 1, grouping the missing gradient with the same sign.
    EXPECT_FLOAT_EQ(split.loss_chg, (missing < 0.0 ? 12.25f : 10.083333f) * this->kSplitTargets);
    EXPECT_EQ(split.DefaultLeft(), missing > 0.0);
    this->CheckCategories({false, true, false});
  }
  auto const &split = this->Evaluate(
      {{{1.0, 0.5}, {-0.5, 0.5}, {0.5, 0.5}}, {{0.5, 0.5}, {1.0, 0.5}, {-0.5, 0.5}}}, {},
      {{"max_cat_to_onehot", "4"},
       {"reg_alpha", "0.1"},
       {"reg_lambda", "1"},
       {"max_delta_step", "0.25"}});
  EXPECT_NEAR(split.loss_chg, this->kSplitTargets == 1 ? 0.3875f : 0.45f, kRtEps);
  EXPECT_EQ(split.split_value, 1.0f);
  this->CheckCategories({false, true, false});
  this->CheckApply();

  // For two categories, one-hot and partition scans find the same split gain.
  this->cuts_ = MakeCutsForTest({0.0, 1.0}, {0, 2}, DeviceOrd::CPU());
  auto onehot = this->Evaluate({{{2.0, 1.0}, {1.0, 1.0}}}, {}, {{"max_cat_to_onehot", "4"}});
  auto const &partition = this->Evaluate({{{2.0, 1.0}, {1.0, 1.0}}});
  EXPECT_EQ(onehot.loss_chg, partition.loss_chg);
}

TYPED_TEST(TestHistEvaluator, PartitionMissing) {
  this->cuts_ = MakeCutsForTest({0.0, 1.0, 2.0, 3.0}, {0, 4}, DeviceOrd::CPU());
  auto const &split =
      this->Evaluate({{{0.5, 0.5}, {0.5, 0.5}, {1.0, 1.0}, {1.0, 1.0}}}, {{-2.0, 3.0}});
  EXPECT_NEAR(split.loss_chg, 2.97619f * this->kSplitTargets, kRtEps * this->kSplitTargets);
  EXPECT_FALSE(split.DefaultLeft());
  EXPECT_TRUE(std::isnan(split.split_value));
  this->CheckCategories({true, false, false, false});
  for (bst_target_t t = 0; t < this->kSplitTargets; ++t) {
    EXPECT_EQ(this->Stats(split.left_sum, t), GradientPairPrecise(2.5, 2.5));
    EXPECT_EQ(this->Stats(split.right_sum, t), GradientPairPrecise(-1.5, 3.5));
  }
  this->CheckApply();

  // Only the second target benefits from this partition, with missing values on the right.
  this->cuts_ = MakeCutsForTest({0.0, 1.0, 2.0}, {0, 3}, DeviceOrd::CPU());
  auto const &mixed = this->Evaluate(
      {{{-3.0, 1.0}, {-3.0, 1.0}, {-3.0, 1.0}}, {{-3.0, 1.0}, {-3.0, 1.0}, {-2.0, 1.0}}},
      {{-3.0, 1.0}, {-2.0, 1.0}});
  EXPECT_FLOAT_EQ(mixed.loss_chg, this->kSplitTargets - 1);
  if constexpr (TestFixture::kSplitTargets > 1) {
    EXPECT_FALSE(mixed.DefaultLeft());
    this->CheckCategories({false, false, true});
    EXPECT_EQ(this->Stats(mixed.right_sum, 0), GradientPairPrecise(-6.0, 2.0));
    EXPECT_EQ(this->Stats(mixed.right_sum, 1), GradientPairPrecise(-4.0, 2.0));
    this->CheckApply();
  }
}

TYPED_TEST(TestHistEvaluator, PartitionParentWeight) {
  // Shift all category gradients to give the parent a positive, zero, or negative weight.
  for (auto offset : {-2.0, 0.0, 2.0}) {
    auto const &split =
        this->Evaluate({{{1.0 + offset, 1.0}, {-2.0 + offset, 1.0}, {1.0 + offset, 1.0}}});
    if (offset == 0.0 && this->kSplitTargets > 1) {
      // With multiple targets the zero projection still preserves bin order.
      EXPECT_FLOAT_EQ(split.loss_chg, 1.5f * this->kSplitTargets);
      this->CheckCategories({true, false, false});
    } else {
      // Isolate category 1; the child orientation can differ.
      EXPECT_FLOAT_EQ(split.loss_chg, 6.0f * this->kSplitTargets);
      common::KCatBitField bits{split.cat_bits};
      EXPECT_EQ(bits.Check(0), bits.Check(2));
      EXPECT_NE(bits.Check(0), bits.Check(1));
    }
  }
}

TYPED_TEST(TestHistEvaluator, PartitionUnregularizedOrder) {
  // -G/H sorts the categories as (1, 2, 0). Sorting by -G/(H + lambda) gives (2, 1, 0) and
  // misses isolating category 1: 2^2/(3+2) + 5^2/(2+2) - 7^2/(5+2) = 0.05.
  auto const &split =
      this->Evaluate({{{-4.0, 1.0}, {-2.0, 3.0}, {-1.0, 1.0}}}, {}, {{"reg_lambda", "2"}});
  EXPECT_NEAR(split.loss_chg, 0.05f * this->kSplitTargets, kRtEps);
  this->CheckCategories({false, true, false});
}
}  // anonymous namespace
}  // namespace xgboost::tree
