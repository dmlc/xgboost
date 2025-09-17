/**
 * Copyright 2022-2024, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>                // for GradientPairInternal, GradientPairPrecise
#include <xgboost/data.h>                // for MetaInfo
#include <xgboost/host_device_vector.h>  // for HostDeviceVector
#include <xgboost/span.h>                // for operator!=, Span, SpanIterator

#include <algorithm>  // for max, max_element, next_permutation, copy
#include <cmath>      // for isnan
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t, uint64_t, uint32_t
#include <limits>     // for numeric_limits
#include <vector>     // for vector

#include "../../../src/common/hist_util.h"      // for HistogramCuts, HistCollection, GHistRow
#include "../../../src/tree/hist/hist_cache.h"  // for HistogramCollection
#include "../../../src/tree/param.h"            // for TrainParam, GradStats

namespace xgboost::tree {
/**
 * @brief Enumerate all possible partitions for categorical split.
 */
class TestPartitionBasedSplit : public ::testing::Test {
 protected:
  size_t n_bins_ = 6;
  std::vector<size_t> sorted_idx_;
  TrainParam param_;
  MetaInfo info_;
  float best_score_{-std::numeric_limits<float>::infinity()};
  common::HistogramCuts cuts_;
  BoundedHistCollection hist_;
  GradientPairPrecise total_gpair_;

  void SetUp() override;
};

inline auto MakeCutsForTest(std::vector<float> values, std::vector<uint32_t> ptrs,
                            std::vector<float> min_values, DeviceOrd device) {
  common::HistogramCuts cuts;
  cuts.cut_values_.HostVector() = values;
  cuts.cut_ptrs_.HostVector() = ptrs;
  cuts.min_vals_.HostVector() = min_values;

  if (device.IsCUDA()) {
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
    cuts_ = MakeCutsForTest({0.0, 1.0, 2.0, 3.0}, {0, 4}, {0.0}, DeviceOrd::CPU());
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
