/**
 * Copyright 2022-2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>                       // for GradientPairInternal, GradientPairPrecise
#include <xgboost/data.h>                       // for MetaInfo
#include <xgboost/host_device_vector.h>         // for HostDeviceVector
#include <xgboost/span.h>                       // for operator!=, Span, SpanIterator

#include <algorithm>                            // for max, max_element, next_permutation, copy
#include <cmath>                                // for isnan
#include <cstddef>                              // for size_t
#include <cstdint>                              // for int32_t, uint64_t, uint32_t
#include <limits>                               // for numeric_limits
#include <numeric>                              // for iota
#include <tuple>                                // for make_tuple, tie, tuple
#include <utility>                              // for pair
#include <vector>                               // for vector

#include "../../../src/common/hist_util.h"      // for HistogramCuts, HistCollection, GHistRow
#include "../../../src/tree/param.h"            // for TrainParam, GradStats
#include "../../../src/tree/split_evaluator.h"  // for TreeEvaluator
#include "../helpers.h"                         // for SimpleLCG, SimpleRealUniformDistribution
#include "gtest/gtest_pred_impl.h"              // for AssertionResult, ASSERT_EQ, ASSERT_TRUE

namespace xgboost::tree {
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
  common::HistCollection hist_;
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

    cuts_.min_vals_.Resize(1);

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

    auto enumerate = [this, n_feat = info_.num_col_](common::GHistRow hist,
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

inline auto MakeCutsForTest(std::vector<float> values, std::vector<uint32_t> ptrs,
                            std::vector<float> min_values, int32_t device) {
  common::HistogramCuts cuts;
  cuts.cut_values_.HostVector() = values;
  cuts.cut_ptrs_.HostVector() = ptrs;
  cuts.min_vals_.HostVector() = min_values;

  if (device >= 0) {
    cuts.cut_ptrs_.SetDevice(device);
    cuts.cut_values_.SetDevice(device);
    cuts.min_vals_.SetDevice(device);
  }

  return cuts;
}

class TestCategoricalSplitWithMissing : public testing::Test {
 protected:
  common::HistogramCuts cuts_;
  // Setup gradients and parent sum with missing values.
  GradientPairPrecise parent_sum_{1.0, 6.0};
  std::vector<GradientPairPrecise> feature_histogram_{
      {0.5, 0.5}, {0.5, 0.5}, {1.0, 1.0}, {1.0, 1.0}};
  TrainParam param_;

  void SetUp() override {
    cuts_ = MakeCutsForTest({0.0, 1.0, 2.0, 3.0}, {0, 4}, {0.0}, -1);
    auto max_cat = *std::max_element(cuts_.cut_values_.HostVector().begin(),
                                     cuts_.cut_values_.HostVector().end());
    cuts_.SetCategorical(true, max_cat);
    param_.UpdateAllowUnknown(
        Args{{"min_child_weight", "0"}, {"reg_lambda", "0"}, {"max_cat_to_onehot", "1"}});
  }

  void CheckResult(float loss_chg, bst_feature_t split_ind, float fvalue, bool is_cat,
                   bool dft_left, GradientPairPrecise left_sum, GradientPairPrecise right_sum) {
    // forward
    // it: 0, gain: 0.545455
    // it: 1, gain: 1.000000
    // it: 2, gain: 2.250000
    // backward
    // it: 3, gain: 1.000000
    // it: 2, gain: 2.250000
    // it: 1, gain: 3.142857
    ASSERT_NEAR(loss_chg, 2.97619, kRtEps);
    ASSERT_TRUE(is_cat);
    ASSERT_TRUE(std::isnan(fvalue));
    ASSERT_EQ(split_ind, 0);
    ASSERT_FALSE(dft_left);
    ASSERT_EQ(left_sum.GetHess(), 2.5);
    ASSERT_EQ(right_sum.GetHess(), parent_sum_.GetHess() - left_sum.GetHess());
  }
};
}  // namespace xgboost::tree
