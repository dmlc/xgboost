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
  dh::device_vector<common::Span<std::uint32_t const>> ridxs{dh::ToSpan(ridx)};

  histogram.AllocateHistograms(&ctx, {0});
  dh::device_vector<common::Span<GradientPairInt64>> hists{histogram.GetNodeHistogram(0)};
  auto quantizers = MakeDummyQuantizers(n_targets);

  histogram.BuildHistogram(ctx.CUDACtx(), page->GetDeviceEllpack(&ctx, {}), fg_acc,
                           gpairs.View(ctx.Device()), dh::ToSpan(ridxs), dh::ToSpan(hists),
                           dh::ToSpan(quantizers));

  common::Span<GradientPairInt64> node_hist = hists[0];
  std::vector<GradientPairInt64> h_node_hist(node_hist.size());
  dh::CopyDeviceSpanToVector(&h_node_hist, node_hist);
  // The values are evenly distributed across all bins
  auto expected = n_samples / n_bins;
  for (auto v : h_node_hist) {
    ASSERT_EQ(v.GetQuantisedGrad(), expected);
    ASSERT_EQ(v.GetQuantisedHess(), expected);
  }
}

TEST(GpuMultiHistogram, MultiNode) {
  auto ctx = MakeCUDACtx(0);
  bst_bin_t n_bins = 16;
  bst_target_t n_targets = 2;
  bst_feature_t n_features = 4;

  bst_idx_t n_samples = 128;
  auto page = MakeEllpackForTest(&ctx, n_samples, n_features, n_bins);

  auto cuts = page->CutsShared();

  FeatureGroups fg{*cuts, true, std::numeric_limits<std::size_t>::max()};
  auto fg_acc = fg.DeviceAccessor(ctx.Device());

  DeviceHistogramBuilder histogram;
  bst_bin_t n_total_bins = n_targets * n_features * n_bins;
  bst_node_t n_nodes = 3;
  histogram.Reset(&ctx, /*max_cached_hist_nodes=*/n_nodes, fg_acc, n_total_bins, true);

  auto gpairs = linalg::Constant(&ctx, GradientPair{1.0f, 1.0f}, n_samples, n_targets);
  auto quantizers = MakeDummyQuantizers(n_targets);

  // Create row indices for multiple nodes (simulating tree partitioning)
  std::vector<dh::device_vector<std::uint32_t>> ridx_vecs(n_nodes);
  std::vector<common::Span<const std::uint32_t>> ridx_spans(n_nodes);

  // Partition rows among nodes
  bst_idx_t rows_per_node = n_samples / n_nodes;
  for (bst_node_t nid = 0; nid < n_nodes; ++nid) {
    bst_idx_t start = nid * rows_per_node;
    bst_idx_t end = (nid == n_nodes - 1) ? n_samples : (nid + 1) * rows_per_node;
    bst_idx_t count = end - start;

    ridx_vecs[nid].resize(count);
    thrust::sequence(ctx.CUDACtx()->CTP(), ridx_vecs[nid].begin(), ridx_vecs[nid].end(), start);
    ridx_spans[nid] = dh::ToSpan(ridx_vecs[nid]);
  }

  // Allocate histograms for all nodes
  std::vector<bst_node_t> node_ids(n_nodes);
  for (bst_node_t nid = 0; nid < n_nodes; ++nid) {
    node_ids[nid] = nid;
  }
  histogram.AllocateHistograms(&ctx, node_ids);

  // Get histogram spans for all nodes
  std::vector<common::Span<GradientPairInt64>> hist_spans(n_nodes);
  for (bst_node_t nid = 0; nid < n_nodes; ++nid) {
    hist_spans[nid] = histogram.GetNodeHistogram(nid);
  }

  // Build histograms for all nodes in one call
  histogram.BuildHistogram(ctx.CUDACtx(), page->GetDeviceEllpack(&ctx, {}), fg_acc,
                           gpairs.View(ctx.Device()), dh::ToSpan(ridx_spans),
                           dh::ToSpan(hist_spans), dh::ToSpan(quantizers));

  // Verify each node's histogram
  for (bst_node_t nid = 0; nid < n_nodes; ++nid) {
    std::vector<GradientPairInt64> h_node_hist(hist_spans[nid].size());
    dh::CopyDeviceSpanToVector(&h_node_hist, hist_spans[nid]);

    bst_idx_t start = nid * rows_per_node;
    bst_idx_t end = (nid == n_nodes - 1) ? n_samples : (nid + 1) * rows_per_node;
    bst_idx_t expected_samples = end - start;

    // The values should be distributed based on the rows in this node
    auto expected = expected_samples / n_bins;
    for (auto v : h_node_hist) {
      ASSERT_EQ(v.GetQuantisedGrad(), expected) << "Node: " << nid;
      ASSERT_EQ(v.GetQuantisedHess(), expected) << "Node: " << nid;
    }
  }
}
}  // namespace xgboost::tree::cuda_impl
