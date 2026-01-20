/**
 * Copyright 2026, XGBoost Contributors
 */
#pragma once
#include <gtest/gtest.h>
#include <xgboost/base.h>    // for bst_idx_t, bst_target_t, GradientPair, GradientPairInt64
#include <xgboost/linalg.h>  // for MatrixView

#include <cstddef>      // for size_t
#include <type_traits>  // for is_same_v

namespace xgboost::tree {
/**
 * @brief Verify that the sampling mask was correctly applied.
 *
 * @tparam SplitGrad  Gradient type for split gradient (GradientPair or GradientPairInt64)
 */
template <typename SplitGrad>
void VerifyApplySamplingMask(linalg::MatrixView<SplitGrad> h_split,
                             linalg::MatrixView<GradientPair> h_value, float subsample) {
  auto n_samples = h_value.Shape(0);
  auto n_value_targets = h_value.Shape(1);

  // Helper to get hessian value from different gradient types
  auto get_hessian = [](auto const& g) {
    using T = std::remove_cv_t<std::remove_reference_t<decltype(g)>>;
    if constexpr (std::is_same_v<T, GradientPairInt64>) {
      return g.GetQuantisedHess();
    } else {
      return g.GetHess();
    }
  };

  std::size_t sampled_count = 0;
  for (bst_idx_t i = 0; i < n_samples; ++i) {
    bool split_is_zero = get_hessian(h_split(i, 0)) == 0;

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
}  // namespace xgboost::tree
