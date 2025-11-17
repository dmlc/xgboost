/**
 * Copyright 2025, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <thrust/sequence.h>

#include "../../../../src/tree/gpu_hist/histogram.cuh"
#include "../../helpers.h"
#include "../../histogram_helpers.h"
#include "dummy_quantizer.cuh"  // for MakeDummyQuantizers

namespace xgboost::tree::cuda_impl {
TEST(GpuMultiHistogram, Basic) {
  auto ctx = MakeCUDACtx(0);
  bst_bin_t n_bins = 16;
  bst_target_t n_targets = 2;
  bst_feature_t n_features = 4;

  bst_idx_t n_samples = 64;
  auto page = MakeEllpackForTest(&ctx, n_samples, n_features, n_bins);

  auto cuts = page->CutsShared();

  FeatureGroups fg{*cuts, true, std::numeric_limits<std::size_t>::max()};
  auto fg_acc = fg.DeviceAccessor(ctx.Device());

  DeviceHistogramBuilder histogram;
  bst_bin_t n_total_bins = n_targets * n_features * n_bins;
  histogram.Reset(&ctx, /*max_cached_hist_nodes=*/2, fg_acc, n_total_bins, true);

  auto gpairs = linalg::Constant(&ctx, GradientPair{1.0f, 1.0f}, n_samples, n_targets);
  dh::device_vector<std::uint32_t> ridx(n_samples);
  thrust::sequence(ctx.CUDACtx()->CTP(), ridx.begin(), ridx.end(), 0);

  histogram.AllocateHistograms(&ctx, {0});
  auto node_hist = histogram.GetNodeHistogram(0);
  auto quantizers = MakeDummyQuantizers(n_targets);

  histogram.BuildHistogram(ctx.CUDACtx(), page->GetDeviceEllpack(&ctx, {}), fg_acc,
                           gpairs.View(ctx.Device()), dh::ToSpan(ridx), node_hist,
                           dh::ToSpan(quantizers));

  std::vector<GradientPairInt64> h_node_hist(node_hist.size());
  dh::CopyDeviceSpanToVector(&h_node_hist, node_hist);
  // The values are evenly distributed across all bins
  auto expected = n_samples / n_bins;
  for (auto v : h_node_hist) {
    ASSERT_EQ(v.GetQuantisedGrad(), expected);
    ASSERT_EQ(v.GetQuantisedHess(), expected);
  }
}
}  // namespace xgboost::tree::cuda_impl
