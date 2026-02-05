/**
 * Copyright 2025-2026, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <thrust/sequence.h>

#include <cuda/functional>

#include "../../../../src/tree/gpu_hist/histogram.cuh"
#include "../../helpers.h"
#include "../../histogram_helpers.h"
#include "dummy_quantizer.cuh"  // for MakeDummyQuantizers

namespace xgboost::tree::cuda_impl {
class MultiHistTest
    : public ::testing::TestWithParam<std::tuple<bst_idx_t, bst_feature_t, bst_target_t, bool>> {
 public:
  Context ctx{MakeCUDACtx(0)};

  bst_bin_t n_bins = 256;

  bst_target_t n_targets{0};
  bst_feature_t n_features{0};

  bst_idx_t n_samples{0};

  std::unique_ptr<EllpackPageImpl> page;

  std::shared_ptr<common::HistogramCuts const> cuts;
  std::unique_ptr<FeatureGroups> p_fg;

  DeviceHistogramBuilder histogram;
  common::Span<GradientPairInt64> node_hist;
  linalg::Matrix<GradientPair> gpairs;
  linalg::Matrix<GradientPairInt64> gpairs_i64;
  dh::device_vector<std::uint32_t> ridx;
  dh::device_vector<GradientQuantiser> quantizers;

  void SetUp() override {
    bool force_global = false;
    std::tie(this->n_samples, this->n_features, this->n_targets, force_global) = this->GetParam();

    this->page = MakeEllpackForTest(&ctx, n_samples, n_features, n_bins);
    this->cuts = page->CutsShared();

    this->p_fg = std::make_unique<FeatureGroups>(*cuts, true, DftMtHistShmemBytes(ctx.Ordinal()));

    this->gpairs = linalg::Constant(&ctx, GradientPair{1.0f, 1.0f}, n_samples, n_targets);
    this->quantizers = MakeDummyQuantizers(n_targets);
    CalcQuantizedGpairs(&this->ctx, this->gpairs.View(this->ctx.Device()),
                        dh::ToSpan(this->quantizers), &gpairs_i64);

    bst_bin_t n_total_bins = n_targets * n_features * n_bins;
    this->histogram.Reset(&ctx, /*max_cached_hist_nodes=*/3, n_total_bins, force_global);

    this->ridx.resize(n_samples);
    thrust::sequence(ctx.CUDACtx()->CTP(), ridx.begin(), ridx.end(), 0);

    this->histogram.AllocateHistograms(&ctx, {0});
    this->node_hist = histogram.GetNodeHistogram(0);
  }

  void TestMtBuild() {
    auto ridxs = dh::device_vector<common::Span<std::uint32_t const>>{dh::ToSpan(ridx)};
    auto hists = dh::device_vector<common::Span<GradientPairInt64>>{node_hist};
    auto sizes_cum = std::vector<std::size_t>{0, ridx.size()};

    this->histogram.BuildHistogram(
        &this->ctx, page->GetDeviceEllpack(&ctx, {}), p_fg->DeviceAccessor(ctx.Device()),
        gpairs_i64.View(this->ctx.Device()), dh::ToSpan(ridxs), dh::ToSpan(hists), sizes_cum);

    auto d_hist = this->node_hist;
    std::vector<GradientPairInt64> h_hist(d_hist.size());
    dh::CopyDeviceSpanToVector(&h_hist, d_hist);
    // The values are evenly distributed across all bins
    auto expected = n_samples / n_bins;
    std::int32_t k = 0;
    for (auto v : h_hist) {
      ASSERT_EQ(v.GetQuantisedGrad(), expected) << " k:" << k;
      ASSERT_EQ(v.GetQuantisedHess(), expected) << " k:" << k;
      ++k;
    }
  }

  void TestMtChildrenBuild() {
    auto d_ridx = dh::ToSpan(ridx);
    auto ridxs = dh::device_vector<common::Span<std::uint32_t const>>{
        d_ridx.subspan(0, n_samples / 4), d_ridx.subspan(n_samples / 4)};
    auto sizes_cum = std::vector<std::size_t>{0, n_samples / 4, n_samples};
    this->histogram.AllocateHistograms(&ctx, {1, 2});
    auto hists = dh::device_vector<common::Span<GradientPairInt64>>{
        this->histogram.GetNodeHistogram(1), this->histogram.GetNodeHistogram(2)};

    this->histogram.BuildHistogram(
        &this->ctx, page->GetDeviceEllpack(&ctx, {}), p_fg->DeviceAccessor(ctx.Device()),
        gpairs_i64.View(this->ctx.Device()), dh::ToSpan(ridxs), dh::ToSpan(hists), sizes_cum);

    auto d_hist_1 = this->histogram.GetNodeHistogram(1);
    auto d_hist_2 = this->histogram.GetNodeHistogram(2);
    std::vector<GradientPairInt64> h_hist_1(d_hist_1.size());
    std::vector<GradientPairInt64> h_hist_2(d_hist_2.size());
    dh::CopyDeviceSpanToVector(&h_hist_1, d_hist_1);
    dh::CopyDeviceSpanToVector(&h_hist_2, d_hist_2);
    ASSERT_EQ(h_hist_1.size(), h_hist_2.size());

    // The values are evenly distributed across all bins
    auto expected = n_samples / n_bins;

    for (std::size_t i = 0; i < h_hist_1.size(); ++i) {
      ASSERT_EQ(h_hist_1[i].GetQuantisedHess() + h_hist_2[i].GetQuantisedHess(), expected)
          << "i:" << i << " l:" << h_hist_1[i].GetQuantisedHess()
          << " r:" << h_hist_2[i].GetQuantisedHess();
    }
  }
};

TEST_P(MultiHistTest, Root) { this->TestMtBuild(); }

TEST_P(MultiHistTest, Children) { this->TestMtChildrenBuild(); }

namespace {
std::string TestName(::testing::TestParamInfo<MultiHistTest::ParamType> const& info) {
  std::stringstream ss;
  auto [n_samples, n_features, n_targets, global] = info.param;
  ss << "n_samples_" << n_samples << "_n_features_" << n_features << "_n_targets_" << n_targets
     << "_global_" << global;
  return ss.str();
}
}  // namespace

INSTANTIATE_TEST_SUITE_P(Basic, MultiHistTest,
                         ::testing::Combine(::testing::Values<bst_idx_t>(256, 1024, 8192),
                                            ::testing::Values(1, 128, 257),
                                            ::testing::Values(1, 16), ::testing::Bool()),
                         TestName);

INSTANTIATE_TEST_SUITE_P(Large, MultiHistTest,
                         ::testing::Combine(::testing::Values<bst_idx_t>((1ul << 21)),
                                            ::testing::Values(2), ::testing::Values(2),
                                            ::testing::Bool()),
                         TestName);
}  // namespace xgboost::tree::cuda_impl
