/**
 * Copyright 2026, XGBoost Contributors
 */
#pragma once
#include <gtest/gtest.h>
#include <xgboost/base.h>    // for bst_idx_t, bst_target_t, GradientPair, GradientPairInt64
#include <xgboost/linalg.h>  // for MatrixView

#include <cstddef>  // for size_t
#include <vector>   // for vector

namespace xgboost::tree {
// Check that multi-target rows are consistently sampled and return count.
inline bst_idx_t CheckSampledRows(linalg::MatrixView<GradientPair const> gpair) {
  auto n_samples = gpair.Shape(0);
  auto n_targets = gpair.Shape(1);
  bst_idx_t sampled_count = 0;
  for (std::size_t i = 0; i < n_samples; ++i) {
    bool first_is_zero = (gpair(i, 0).GetGrad() == 0.0f && gpair(i, 0).GetHess() == 0.0f);
    for (bst_target_t t = 1; t < n_targets; ++t) {
      bool is_zero = (gpair(i, t).GetGrad() == 0.0f && gpair(i, t).GetHess() == 0.0f);
      EXPECT_EQ(first_is_zero, is_zero);
    }
    if (!first_is_zero) {
      ++sampled_count;
    }
  }
  return sampled_count;
}

// Check that sampling mask was correctly applied from split gradient to value gradient.
inline void CheckSamplingMask(linalg::MatrixView<GradientPair> h_split,
                              linalg::MatrixView<GradientPair> h_value, float subsample) {
  auto n_samples = h_value.Shape(0);
  auto n_value_targets = h_value.Shape(1);

  std::size_t sampled_count = 0;
  for (bst_idx_t i = 0; i < n_samples; ++i) {
    bool split_is_zero = h_split(i, 0).GetHess() == 0;

    for (bst_target_t t = 0; t < n_value_targets; ++t) {
      bool value_is_zero = (h_value(i, t).GetGrad() == 0.0f && h_value(i, t).GetHess() == 0.0f);
      ASSERT_EQ(split_is_zero, value_is_zero);
    }

    if (!split_is_zero) {
      ++sampled_count;
    }
  }

  // Verify approximately the right fraction of rows are sampled
  double sampled_fraction = static_cast<double>(sampled_count) / static_cast<double>(n_samples);
  ASSERT_NEAR(sampled_fraction, subsample, 0.05);
}

inline void CheckSampling(float subsample, bst_target_t n_targets, bool check_sum,
                          std::vector<GradientPairPrecise> const& sum_sampled_gpair,
                          std::vector<GradientPairPrecise> const& sum_gpair,
                          linalg::MatrixView<GradientPair> h_gpair) {
  auto n_samples = h_gpair.Shape(0);
  bst_idx_t sample_rows = n_samples * subsample;

  // Verify gradient sums per target
  for (bst_target_t t = 0; t < n_targets; ++t) {
    if (check_sum) {
      // Gradient-based sampling preserves the sum approximately
      ASSERT_NEAR(sum_gpair[t].GetGrad(), sum_sampled_gpair[t].GetGrad(), 0.03f * n_samples);
      ASSERT_NEAR(sum_gpair[t].GetHess(), sum_sampled_gpair[t].GetHess(), 0.03f * n_samples);
    } else {
      // Uniform sampling preserves the mean approximately
      auto mean_grad = sum_gpair[t].GetGrad() / n_samples;
      auto mean_hess = sum_gpair[t].GetHess() / n_samples;
      auto sampled_mean_grad = sum_sampled_gpair[t].GetGrad() / sample_rows;
      auto sampled_mean_hess = sum_sampled_gpair[t].GetHess() / sample_rows;
      ASSERT_NEAR(mean_grad, sampled_mean_grad, mean_grad * 0.1);
      ASSERT_NEAR(mean_hess, sampled_mean_hess, mean_hess * 0.1);
    }
  }

  // Verify multi-target consistency and sample fraction (reuse CheckSampledRows)
  auto sampled_count = CheckSampledRows(h_gpair);
  if (subsample < 1.0f) {
    double sampled_fraction = static_cast<double>(sampled_count) / n_samples;
    ASSERT_NEAR(sampled_fraction, subsample, 0.05);
  }
}
}  // namespace xgboost::tree
