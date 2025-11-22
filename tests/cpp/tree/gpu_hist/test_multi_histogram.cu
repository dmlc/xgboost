/**
 * Copyright 2025, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <thrust/sequence.h>

#include <cuda/functional>

#include "../../../../src/common/device_debug.cuh"
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

  FeatureGroups fg{*cuts, n_targets, true, dh::MaxSharedMemory(ctx.Ordinal())};
  auto fg_acc = fg.DeviceAccessor(ctx.Device());

  DeviceHistogramBuilder histogram;
  bst_bin_t n_total_bins = n_targets * n_features * n_bins;
  histogram.Reset(&ctx, /*max_cached_hist_nodes=*/2, fg_acc, n_total_bins, false);

  auto gpairs = linalg::Constant(&ctx, GradientPair{1.0f, 1.0f}, n_samples, n_targets);
  dh::device_vector<std::uint32_t> ridx(n_samples);
  thrust::sequence(ctx.CUDACtx()->CTP(), ridx.begin(), ridx.end(), 0);

  histogram.AllocateHistograms(&ctx, {0});
  auto node_hist = histogram.GetNodeHistogram(0);
  auto quantizers = MakeDummyQuantizers(n_targets);

  dh::device_vector<common::Span<std::uint32_t const>> ridxs{dh::ToSpan(ridx)};
  dh::device_vector<common::Span<GradientPairInt64>> hists{node_hist};
  histogram.BuildHistogram(ctx.CUDACtx(), page->GetDeviceEllpack(&ctx, {}), fg_acc,
                           gpairs.View(ctx.Device()), dh::ToSpan(ridxs), dh::ToSpan(hists),
                           ridx.size(), dh::ToSpan(quantizers));

  std::vector<GradientPairInt64> h_node_hist(node_hist.size());
  dh::CopyDeviceSpanToVector(&h_node_hist, node_hist);
  // The values are evenly distributed across all bins
  auto expected = n_samples / n_bins;
  for (auto v : h_node_hist) {
    EXPECT_EQ(v.GetQuantisedGrad(), expected);
    EXPECT_EQ(v.GetQuantisedHess(), expected);
  }
}

class MultiHistTest
    : public ::testing::TestWithParam<std::tuple<bst_idx_t, bst_feature_t, bst_target_t>> {
 public:
  Context ctx{MakeCUDACtx(0)};

  bst_bin_t n_bins = 256;

  bst_target_t n_targets{0};
  bst_feature_t n_features{0};

  bool use_single_target = false;

  bst_idx_t n_samples{0};

  std::unique_ptr<EllpackPageImpl> page;

  std::shared_ptr<common::HistogramCuts const> cuts;
  std::unique_ptr<FeatureGroups> p_fg;

  DeviceHistogramBuilder histogram;
  common::Span<GradientPairInt64> node_hist;
  linalg::Matrix<GradientPair> gpairs;
  dh::device_vector<std::uint32_t> ridx;
  dh::device_vector<GradientQuantiser> quantizers;

  void SetUp() override {
    std::tie(this->n_samples, this->n_features, this->n_targets) = this->GetParam();

    this->page = MakeEllpackForTest(&ctx, n_samples, n_features, n_bins);
    this->cuts = page->CutsShared();

    this->p_fg =
        std::make_unique<FeatureGroups>(*cuts, this->n_targets, true, dh::MaxSharedMemory(0));

    bst_bin_t n_total_bins = n_targets * n_features * n_bins;
    auto fg_acc = p_fg->DeviceAccessor(ctx.Device());
    this->histogram.Reset(&ctx, /*max_cached_hist_nodes=*/3, fg_acc, n_total_bins, false);

    this->gpairs = linalg::Constant(&ctx, GradientPair{1.0f, 1.0f}, n_samples, n_targets);

    this->ridx.resize(n_samples);
    thrust::sequence(ctx.CUDACtx()->CTP(), ridx.begin(), ridx.end(), 0);

    this->histogram.AllocateHistograms(&ctx, {0});
    this->node_hist = histogram.GetNodeHistogram(0);

    this->quantizers = MakeDummyQuantizers(n_targets);
  }

  void TestMtBuild() {
    auto ridxs = dh::device_vector<common::Span<std::uint32_t const>>{dh::ToSpan(ridx)};
    auto hists = dh::device_vector<common::Span<GradientPairInt64>>{node_hist};
    this->histogram.BuildHistogram(this->ctx.CUDACtx(), page->GetDeviceEllpack(&ctx, {}),
                                   p_fg->DeviceAccessor(ctx.Device()),
                                   this->gpairs.View(this->ctx.Device()), dh::ToSpan(ridxs),
                                   dh::ToSpan(hists), ridx.size(), dh::ToSpan(this->quantizers));

    std::vector<GradientPairInt64> h_node_hist(node_hist.size());
    dh::CopyDeviceSpanToVector(&h_node_hist, node_hist);
    // The values are evenly distributed across all bins
    auto expected = n_samples / n_bins;
    std::int32_t k = 0;
    for (auto v : h_node_hist) {
      ASSERT_EQ(v.GetQuantisedGrad(), expected) << " k:" << k;
      ASSERT_EQ(v.GetQuantisedHess(), expected) << " k:" << k;
      ++k;
    }
  }

  void TestMtChildrenBuild() {
    auto d_ridx = dh::ToSpan(ridx);
    auto ridxs = dh::device_vector<common::Span<std::uint32_t const>>{
        d_ridx.subspan(0, n_samples / 4), d_ridx.subspan(n_samples / 4)};
    this->histogram.AllocateHistograms(&ctx, {1, 2});
    auto hists = dh::device_vector<common::Span<GradientPairInt64>>{
        this->histogram.GetNodeHistogram(1), this->histogram.GetNodeHistogram(2)};
    this->histogram.BuildHistogram(this->ctx.CUDACtx(), page->GetDeviceEllpack(&ctx, {}),
                                   p_fg->DeviceAccessor(ctx.Device()),
                                   this->gpairs.View(this->ctx.Device()), dh::ToSpan(ridxs),
                                   dh::ToSpan(hists), ridx.size(), dh::ToSpan(this->quantizers));
  }

  void TestStBuild() {
    GradientQuantiser q{GradientPairPrecise{1.0f, 1.0f}, GradientPairPrecise{1.0f, 1.0f}};
    this->histogram.BuildHistogram(
        this->ctx.CUDACtx(), page->GetDeviceEllpack(&ctx, {}), p_fg->DeviceAccessor(ctx.Device()),
        this->gpairs.Data()->ConstDeviceSpan(), dh::ToSpan(ridx), this->node_hist, q);
  }
};

TEST_P(MultiHistTest, Root) { this->TestMtBuild(); }

TEST_P(MultiHistTest, Children) { this->TestMtChildrenBuild(); }

namespace {
std::string TestName(::testing::TestParamInfo<MultiHistTest::ParamType> const& info) {
  std::stringstream ss;
  auto [n_samples, n_features, n_targets] = info.param;
  ss << "n_samples_" << n_samples << "_n_features_" << n_features << "_n_targets_" << n_targets;
  return ss.str();
}
}  // namespace

INSTANTIATE_TEST_SUITE_P(Histogram, MultiHistTest,
                         ::testing::Combine(::testing::Values(256, 1024, 8192),
                                            ::testing::Values(1, 128, 257),
                                            ::testing::Values(1, 16)),
                         TestName);
}  // namespace xgboost::tree::cuda_impl
