/*!
 * Copyright 2022 by XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <algorithm>  // next_permutation
#include <numeric>    // iota

#include "../../../src/tree/hist/evaluate_splits.h"
#include "../helpers.h"

namespace xgboost {
namespace tree {
/**
 * \brief Enumerate all possible partitions for categorical split.
 */
class TestPartitionBasedSplit : public ::testing::Test {
 protected:
  size_t n_bins_ = 6;
  std::vector<size_t> sorted_idx_;
  TrainParam param_;
  MetaInfo info_;
  float best_score_{-std::numeric_limits<float>::infinity()};
  common::HistogramCuts cuts_;
  common::HistCollection<double> hist_;
  GradientPairPrecise total_gpair_;

  void SetUp() override {
    param_.UpdateAllowUnknown(Args{{"min_child_weight", "0"}, {"reg_lambda", "0"}});
    sorted_idx_.resize(n_bins_);
    std::iota(sorted_idx_.begin(), sorted_idx_.end(), 0);

    info_.num_col_ = 1;

    cuts_.cut_ptrs_.Resize(2);
    cuts_.SetCategorical(true, n_bins_);
    auto &h_cuts = cuts_.cut_ptrs_.HostVector();
    h_cuts[0] = 0;
    h_cuts[1] = n_bins_;
    auto &h_vals = cuts_.cut_values_.HostVector();
    h_vals.resize(n_bins_);
    std::iota(h_vals.begin(), h_vals.end(), 0.0);

    hist_.Init(cuts_.TotalBins());
    hist_.AddHistRow(0);
    hist_.AllocateAllData();
    auto node_hist = hist_[0];

    SimpleLCG lcg;
    SimpleRealUniformDistribution<double> grad_dist{-4.0, 4.0};
    SimpleRealUniformDistribution<double> hess_dist{0.0, 4.0};

    for (auto &e : node_hist) {
      e = GradientPairPrecise{grad_dist(&lcg), hess_dist(&lcg)};
      total_gpair_ += e;
    }

    auto enumerate = [this, n_feat = info_.num_col_](common::GHistRow<double> hist,
                                                     GradientPairPrecise parent_sum) {
      int32_t best_thresh = -1;
      float best_score{-std::numeric_limits<float>::infinity()};
      TreeEvaluator evaluator{param_, static_cast<bst_feature_t>(n_feat), -1};
      auto tree_evaluator = evaluator.GetEvaluator<TrainParam>();
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
      int32_t thresh;
      float score;
      std::vector<GradientPairPrecise> sorted_hist(node_hist.size());
      for (size_t i = 0; i < sorted_hist.size(); ++i) {
        sorted_hist[i] = node_hist[sorted_idx_[i]];
      }
      std::tie(thresh, score) = enumerate({sorted_hist}, total_gpair_);
      if (score > best_score_) {
        best_score_ = score;
      }
    } while (std::next_permutation(sorted_idx_.begin(), sorted_idx_.end()));
  }
};
}  // namespace tree
}  // namespace xgboost
