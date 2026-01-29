/**
 * Copyright 2026, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>

#include <vector>

#include "histogram_helpers.h"

namespace xgboost {
namespace {
// Count occurrences of each bin for each feature
template <typename Accessor>
auto CountBins(Accessor const& accessor, bst_bin_t n_bins_per_feat) {
  auto n_samples = accessor.NumRows();
  auto n_features = accessor.NumFeatures();
  std::vector<std::vector<bst_idx_t>> bin_counts(n_features,
                                                 std::vector<bst_idx_t>(n_bins_per_feat, 0));

  // Count occurrences of each bin for each feature
  for (bst_idx_t row = 0; row < n_samples; ++row) {
    for (bst_feature_t feat = 0; feat < n_features; ++feat) {
      bst_idx_t idx = row * accessor.row_stride + feat;
      bst_bin_t bin = accessor.gidx_iter[idx];

      // The bin values are already local to each feature
      EXPECT_GE(bin, 0);
      EXPECT_LT(bin, n_bins_per_feat);
      bin_counts[feat][bin]++;
    }
  }
  return bin_counts;
}
}  // namespace

TEST(HistogramHelpers, MakeEllpack) {
  auto ctx = MakeCUDACtx(0);

  bst_idx_t n_samples = 100;
  bst_feature_t n_features = 5;
  bst_bin_t n_bins_per_feat = 10;

  auto ellpack = MakeEllpackForTest(&ctx, n_samples, n_features, n_bins_per_feat);

  ASSERT_NE(ellpack, nullptr);
  EXPECT_EQ(ellpack->Size(), n_samples);
  EXPECT_EQ(ellpack->Cuts().NumFeatures(), n_features);

  // Test histogram cuts structure
  const auto& cuts = ellpack->Cuts();
  EXPECT_EQ(cuts.NumFeatures(), n_features);
  EXPECT_EQ(cuts.TotalBins(), n_features * n_bins_per_feat);

  // Verify cut pointers are correct
  const auto& cut_ptrs = cuts.Ptrs();
  EXPECT_EQ(cut_ptrs.size(), n_features + 1);
  for (bst_feature_t f = 0; f < n_features; ++f) {
    EXPECT_EQ(cut_ptrs[f + 1] - cut_ptrs[f], n_bins_per_feat);
  }

  EXPECT_TRUE(ellpack->IsDense());

  std::vector<common::CompressedByteT> h_gidx_buffer;
  auto accessor_var = ellpack->GetHostEllpack(&ctx, &h_gidx_buffer);
  std::visit(
      [&](auto&& accessor) {
        EXPECT_EQ(accessor.row_stride, n_features);
        EXPECT_EQ(accessor.n_rows, n_samples);

        auto bin_counts = CountBins(accessor, n_bins_per_feat);
        // Validate histogram index distribution
        auto n_values_per_bin = n_samples / n_bins_per_feat;
        auto remainder = n_samples % n_bins_per_feat;

        // Verify expected distribution
        for (bst_feature_t feat = 0; feat < n_features; ++feat) {
          for (bst_bin_t bin = 0; bin < n_bins_per_feat; ++bin) {
            bst_idx_t expected_count = n_values_per_bin;
            if (bin == n_bins_per_feat - 1) {
              expected_count += remainder;  // Last bin gets the remainder
            }
            EXPECT_EQ(bin_counts[feat][bin], expected_count);
          }
        }
      },
      accessor_var);
}
}  // namespace xgboost
