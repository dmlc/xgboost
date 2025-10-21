/**
 * Copyright 2025, XGBoost contributors
 */
#include <xgboost/base.h>  // for bst_feature_t
#include <xgboost/data.h>  // for FeatureType
#include <xgboost/span.h>  // for Span

#include <memory>  // for make_unique
#include <random>  // for uniform_real_distribution
#include <vector>  // for vector

#include "../../src/common/device_vector.cuh"  // for device_vector
#include "../../src/common/hist_util.h"        // for HistogramCuts
#include "../../src/data/device_adapter.cuh"   // for CupyAdapter, GetRowCounts
#include "../../src/data/ellpack_page.cuh"     // for EllpackPageImpl
#include "histogram_helpers.h"

namespace xgboost {
[[nodiscard]] std::unique_ptr<EllpackPageImpl> MakeEllpackForTest(Context const* ctx,
                                                                  bst_idx_t n_samples,
                                                                  bst_feature_t n_features,
                                                                  bst_bin_t n_bins_per_feat) {
  // Construct the histogram bins
  std::vector<std::uint32_t> cut_indptr(n_features + 1, 0);
  for (std::size_t i = 1; i < cut_indptr.size(); ++i) {
    cut_indptr[i] = i * n_bins_per_feat;
  }
  std::vector<float> cut_values;
  for (bst_feature_t f_idx = 0; f_idx < n_features; ++f_idx) {
    for (bst_bin_t bin_idx = 0; bin_idx < n_bins_per_feat; ++bin_idx) {
      cut_values.push_back(bin_idx + 1.0f);
    }
  }
  std::vector<float> min_values;
  std::default_random_engine rng(2025);
  std::uniform_real_distribution<float> min_dist(-1.0, -0.5);
  for (bst_feature_t f_idx = 0; f_idx < n_features; ++f_idx) {
    min_values.push_back(min_dist(rng));
  }
  auto p_cuts = std::make_shared<common::HistogramCuts>();
  p_cuts->cut_ptrs_.HostVector() = cut_indptr;
  p_cuts->cut_values_.HostVector() = cut_values;
  p_cuts->min_vals_.HostVector() = min_values;

  // Construct the data
  auto n_values_per_bin = n_samples / n_bins_per_feat;

  linalg::Matrix<float> values{
      {n_samples, static_cast<bst_idx_t>(n_features)}, DeviceOrd::CPU(), linalg::kF};
  auto& h_values = values.Data()->HostVector();
  h_values.clear();

  for (bst_feature_t f_idx = 0; f_idx < n_features; ++f_idx) {
    for (bst_bin_t bin_idx = 0; bin_idx < n_bins_per_feat; ++bin_idx) {
      // min-max value for the current bin
      auto min_value = static_cast<float>(bin_idx + kRtEps);
      auto max_value = static_cast<float>(bin_idx + 1.0 - kRtEps);
      std::uniform_real_distribution<float> dist(min_value, max_value);
      for (std::size_t i = 0; i < n_values_per_bin; ++i) {
        h_values.emplace_back(dist(rng));
      }
      if (bin_idx == n_bins_per_feat - 1) {
        auto remainer = n_samples % n_bins_per_feat;
        for (std::size_t i = 0; i < remainer; ++i) {
          h_values.emplace_back(dist(rng));
        }
      }
    }
  }
  CHECK_EQ(h_values.size(), n_samples * n_features);

  auto str = linalg::ArrayInterfaceStr(values.View(ctx->Device()));
  auto adapter = data::CupyAdapter{StringView{str}};
  dh::device_vector<bst_idx_t> row_counts(n_samples);
  auto missing = std::numeric_limits<float>::quiet_NaN();
  bst_idx_t row_stride =
      GetRowCounts(ctx, adapter.Value(), dh::ToSpan(row_counts), ctx->Device(), missing);
  auto ellpack = std::make_unique<EllpackPageImpl>(
      ctx, adapter.Value(), missing, true, dh::ToSpan(row_counts),
      common::Span<FeatureType const>{}, row_stride, n_samples, p_cuts);

  return ellpack;
}
}  // namespace xgboost
