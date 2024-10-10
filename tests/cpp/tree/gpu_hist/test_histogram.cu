/**
 * Copyright 2020-2024, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>  // for Context

#include <memory>  // for unique_ptr
#include <vector>  // for vector

#include "../../../../src/tree/gpu_hist/histogram.cuh"
#include "../../../../src/tree/gpu_hist/row_partitioner.cuh"  // for RowPartitioner
#include "../../../../src/tree/hist/param.h"                  // for HistMakerTrainParam
#include "../../../../src/tree/param.h"                       // for TrainParam
#include "../../categorical_helpers.h"                        // for OneHotEncodeFeature
#include "../../helpers.h"
#include "../../histogram_helpers.h"  // for BuildEllpackPage

namespace xgboost::tree {
TEST(Histogram, DeviceHistogramStorage) {
  // Ensures that node allocates correctly after reaching `kStopGrowingSize`.
  auto ctx = MakeCUDACtx(0);
  constexpr size_t kNBins = 128;
  constexpr int kNNodes = 4;
  constexpr size_t kStopGrowing = kNNodes * kNBins * 2u;
  DeviceHistogramStorage histogram{};
  histogram.Reset(&ctx, kNBins, kNNodes);
  for (int i = 0; i < kNNodes; ++i) {
    histogram.AllocateHistograms(&ctx, {i});
  }
  ASSERT_EQ(histogram.Data().size(), kStopGrowing);
  histogram.Reset(&ctx, kNBins, kNNodes);

  // Use allocated memory but do not erase nidx_map.
  for (int i = 0; i < kNNodes; ++i) {
    histogram.AllocateHistograms(&ctx, {i});
  }
  for (int i = 0; i < kNNodes; ++i) {
    ASSERT_TRUE(histogram.HistogramExists(i));
  }

  // Add two new nodes
  histogram.AllocateHistograms(&ctx, {kNNodes});
  histogram.AllocateHistograms(&ctx, {kNNodes + 1});

  // Old cached nodes should still exist
  for (int i = 0; i < kNNodes; ++i) {
    ASSERT_TRUE(histogram.HistogramExists(i));
  }

  // Should be deleted
  ASSERT_FALSE(histogram.HistogramExists(kNNodes));
  // Most recent node should exist
  ASSERT_TRUE(histogram.HistogramExists(kNNodes + 1));

  // Add same node again - should fail
  EXPECT_ANY_THROW(histogram.AllocateHistograms(&ctx, {kNNodes + 1}););
}

TEST(Histogram, SubtractionTrack) {
  auto ctx = MakeCUDACtx(0);

  auto page = BuildEllpackPage(&ctx, 64, 4);
  auto cuts = page->CutsShared();
  FeatureGroups fg{*cuts, true, std::numeric_limits<std::size_t>::max()};
  auto fg_acc = fg.DeviceAccessor(ctx.Device());
  auto n_total_bins = cuts->TotalBins();

  // 2 nodes
  auto max_cached_hist_nodes = 2ull;
  DeviceHistogramBuilder histogram;
  histogram.Reset(&ctx, max_cached_hist_nodes, fg_acc, n_total_bins, false);
  histogram.AllocateHistograms(&ctx, {0, 1, 2});
  GPUExpandEntry root;
  root.nid = 0;
  auto need_build = histogram.SubtractHist(&ctx, {root}, {0}, {1});

  std::vector<GPUExpandEntry> candidates(2);
  candidates[0].nid = 1;
  candidates[1].nid = 2;

  need_build = histogram.SubtractHist(&ctx, candidates, {3, 5}, {4, 6});
  ASSERT_EQ(need_build.size(), 2);
  ASSERT_EQ(need_build[0], 4);
  ASSERT_EQ(need_build[1], 6);
}

std::vector<GradientPairPrecise> GetHostHistGpair() {
  // 24 bins, 3 bins for each feature (column).
  std::vector<GradientPairPrecise> hist_gpair = {
    {0.8314f, 0.7147f}, {1.7989f, 3.7312f}, {3.3846f, 3.4598f},
    {2.9277f, 3.5886f}, {1.8429f, 2.4152f}, {1.2443f, 1.9019f},
    {1.6380f, 2.9174f}, {1.5657f, 2.5107f}, {2.8111f, 2.4776f},
    {2.1322f, 3.0651f}, {3.2927f, 3.8540f}, {0.5899f, 0.9866f},
    {1.5185f, 1.6263f}, {2.0686f, 3.1844f}, {2.4278f, 3.0950f},
    {1.5105f, 2.1403f}, {2.6922f, 4.2217f}, {1.8122f, 1.5437f},
    {0.0000f, 0.0000f}, {4.3245f, 5.7955f}, {1.6903f, 2.1103f},
    {2.4012f, 4.4754f}, {3.6136f, 3.4303f}, {0.0000f, 0.0000f}
  };
  return hist_gpair;
}

void TestBuildHist(bool use_shared_memory_histograms) {
  int const kNRows = 16, kNCols = 8;
  auto ctx = MakeCUDACtx(0);

  auto page = BuildEllpackPage(&ctx, kNRows, kNCols);

  xgboost::SimpleLCG gen;
  xgboost::SimpleRealUniformDistribution<bst_float> dist(0.0f, 1.0f);
  HostDeviceVector<GradientPair> gpair(kNRows);
  for (auto& gp : gpair.HostVector()) {
    float grad = dist(&gen);
    float hess = dist(&gen);
    gp = GradientPair{grad, hess};
  }
  gpair.SetDevice(ctx.Device());

  auto row_partitioner = std::make_unique<RowPartitioner>();
  row_partitioner->Reset(&ctx, kNRows, 0);

  auto quantiser = std::make_unique<GradientQuantiser>(&ctx, gpair.ConstDeviceSpan(), MetaInfo());
  auto shm_size = use_shared_memory_histograms ? dh::MaxSharedMemoryOptin(ctx.Ordinal()) : 0;
  FeatureGroups feature_groups(page->Cuts(), page->IsDenseCompressed(), shm_size);

  DeviceHistogramBuilder builder;
  builder.Reset(&ctx, HistMakerTrainParam::CudaDefaultNodes(),
                feature_groups.DeviceAccessor(ctx.Device()), page->Cuts().TotalBins(),
                !use_shared_memory_histograms);
  builder.AllocateHistograms(&ctx, {0});
  builder.BuildHistogram(ctx.CUDACtx(), page->GetDeviceAccessor(&ctx),
                         feature_groups.DeviceAccessor(ctx.Device()), gpair.DeviceSpan(),
                         row_partitioner->GetRows(0), builder.GetNodeHistogram(0), *quantiser);

  auto node_histogram = builder.GetNodeHistogram(0);

  std::vector<GradientPairInt64> h_result(node_histogram.size());
  dh::CopyDeviceSpanToVector(&h_result, node_histogram);

  std::vector<GradientPairPrecise> solution = GetHostHistGpair();
  for (size_t i = 0; i < h_result.size(); ++i) {
    auto result = quantiser->ToFloatingPoint(h_result[i]);
    ASSERT_NEAR(result.GetGrad(), solution[i].GetGrad(), 0.01f);
    ASSERT_NEAR(result.GetHess(), solution[i].GetHess(), 0.01f);
  }
}

TEST(Histogram, BuildHistGlobalMem) {
  TestBuildHist(false);
}

TEST(Histogram, BuildHistSharedMem) {
  TestBuildHist(true);
}

namespace {
void TestDeterministicHistogram(bool is_dense, std::size_t shm_size, bool force_global) {
  Context ctx = MakeCUDACtx(0);
  size_t constexpr kBins = 256, kCols = 120, kRows = 16384, kRounds = 16;
  float constexpr kLower = -1e-2, kUpper = 1e2;

  float sparsity = is_dense ? 0.0f : 0.5f;
  auto matrix = RandomDataGenerator(kRows, kCols, sparsity).GenerateDMatrix();
  auto batch_param = BatchParam{kBins, tree::TrainParam::DftSparseThreshold()};

  for (auto const& batch : matrix->GetBatches<EllpackPage>(&ctx, batch_param)) {
    auto* page = batch.Impl();

    tree::RowPartitioner row_partitioner;
    row_partitioner.Reset(&ctx, kRows, page->base_rowid);
    auto ridx = row_partitioner.GetRows(0);

    bst_bin_t num_bins = kBins * kCols;
    dh::device_vector<GradientPairInt64> histogram(num_bins);
    auto d_histogram = dh::ToSpan(histogram);
    auto gpair = GenerateRandomGradients(kRows, kLower, kUpper);
    gpair.SetDevice(ctx.Device());

    FeatureGroups feature_groups{page->Cuts(), page->IsDenseCompressed(), shm_size};

    auto quantiser = GradientQuantiser(&ctx, gpair.DeviceSpan(), MetaInfo());
    DeviceHistogramBuilder builder;
    builder.Reset(&ctx, HistMakerTrainParam::CudaDefaultNodes(),
                  feature_groups.DeviceAccessor(ctx.Device()), num_bins, force_global);
    builder.BuildHistogram(ctx.CUDACtx(), page->GetDeviceAccessor(&ctx),
                           feature_groups.DeviceAccessor(ctx.Device()), gpair.DeviceSpan(), ridx,
                           d_histogram, quantiser);

    std::vector<GradientPairInt64> histogram_h(num_bins);
    dh::safe_cuda(cudaMemcpy(histogram_h.data(), d_histogram.data(),
                             num_bins * sizeof(GradientPairInt64), cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < kRounds; ++i) {
      dh::device_vector<GradientPairInt64> new_histogram(num_bins);
      auto d_new_histogram = dh::ToSpan(new_histogram);

      auto quantiser = GradientQuantiser(&ctx, gpair.DeviceSpan(), MetaInfo());
      DeviceHistogramBuilder builder;
      builder.Reset(&ctx, HistMakerTrainParam::CudaDefaultNodes(),
                    feature_groups.DeviceAccessor(ctx.Device()), num_bins, force_global);
      builder.BuildHistogram(ctx.CUDACtx(), page->GetDeviceAccessor(&ctx),
                             feature_groups.DeviceAccessor(ctx.Device()), gpair.DeviceSpan(), ridx,
                             d_new_histogram, quantiser);

      std::vector<GradientPairInt64> new_histogram_h(num_bins);
      dh::safe_cuda(cudaMemcpy(new_histogram_h.data(), d_new_histogram.data(),
                               num_bins * sizeof(GradientPairInt64), cudaMemcpyDeviceToHost));
      for (size_t j = 0; j < new_histogram_h.size(); ++j) {
        ASSERT_EQ(new_histogram_h[j].GetQuantisedGrad(), histogram_h[j].GetQuantisedGrad());
        ASSERT_EQ(new_histogram_h[j].GetQuantisedHess(), histogram_h[j].GetQuantisedHess());
      }
    }

    {
      auto gpair = GenerateRandomGradients(kRows, kLower, kUpper);
      gpair.SetDevice(ctx.Device());

      // Use a single feature group to compute the baseline.
      FeatureGroups single_group(page->Cuts());

      dh::device_vector<GradientPairInt64> baseline(num_bins);
      DeviceHistogramBuilder builder;
      // Single group must use global memory.
      builder.Reset(&ctx, HistMakerTrainParam::CudaDefaultNodes(),
                    single_group.DeviceAccessor(ctx.Device()), num_bins, /*force_global=*/true);
      builder.BuildHistogram(ctx.CUDACtx(), page->GetDeviceAccessor(&ctx),
                             single_group.DeviceAccessor(ctx.Device()), gpair.DeviceSpan(), ridx,
                             dh::ToSpan(baseline), quantiser);

      std::vector<GradientPairInt64> baseline_h(num_bins);
      dh::safe_cuda(cudaMemcpy(baseline_h.data(), baseline.data().get(),
                               num_bins * sizeof(GradientPairInt64), cudaMemcpyDeviceToHost));

      for (size_t i = 0; i < baseline.size(); ++i) {
        ASSERT_NEAR(baseline_h[i].GetQuantisedGrad(), histogram_h[i].GetQuantisedGrad(),
                    baseline_h[i].GetQuantisedGrad() * 1e-3);
      }
    }
  }
}

class TestGPUDeterministic : public ::testing::TestWithParam<std::tuple<bool, std::size_t, bool>> {
 protected:
  void Run() {
    auto [is_dense, shm_size, force_global] = this->GetParam();
    if (shm_size > dh::MaxSharedMemoryOptin(0) && !force_global) {
      force_global = true;  // We will have to skip this test to avoid false check in the builder.
    }
    TestDeterministicHistogram(is_dense, shm_size, force_global);
  }
};
}  // anonymous namespace

TEST_P(TestGPUDeterministic, Histogram) { this->Run(); }

INSTANTIATE_TEST_SUITE_P(Histogram, TestGPUDeterministic,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Values(48 * 1024, 64 * 1024, 160 * 1024),
                                            ::testing::Bool()));

void ValidateCategoricalHistogram(size_t n_categories, common::Span<GradientPairInt64> onehot,
                                  common::Span<GradientPairInt64> cat) {
  auto cat_sum = std::accumulate(cat.cbegin(), cat.cend(), GradientPairInt64{});
  for (size_t c = 0; c < n_categories; ++c) {
    auto zero = onehot[c * 2];
    auto one = onehot[c * 2 + 1];

    auto chosen = cat[c];
    auto not_chosen = cat_sum - chosen;
    ASSERT_EQ(zero, not_chosen);
    ASSERT_EQ(one, chosen);
  }
}

// Test 1 vs rest categorical histogram is equivalent to one hot encoded data.
void TestGPUHistogramCategorical(size_t num_categories) {
  auto ctx = MakeCUDACtx(0);
  size_t constexpr kRows = 340;
  size_t constexpr kBins = 256;
  auto x = GenerateRandomCategoricalSingleColumn(kRows, num_categories);
  auto cat_m = GetDMatrixFromData(x, kRows, 1);
  cat_m->Info().feature_types.HostVector().push_back(FeatureType::kCategorical);
  auto batch_param = BatchParam{kBins, tree::TrainParam::DftSparseThreshold()};
  tree::RowPartitioner row_partitioner;
  row_partitioner.Reset(&ctx, kRows, 0);
  auto ridx = row_partitioner.GetRows(0);
  dh::device_vector<GradientPairInt64> cat_hist(num_categories);
  auto gpair = GenerateRandomGradients(kRows, 0, 2);
  gpair.SetDevice(DeviceOrd::CUDA(0));
  auto quantiser = GradientQuantiser(&ctx, gpair.DeviceSpan(), MetaInfo());
  /**
   * Generate hist with cat data.
   */
  for (auto const &batch : cat_m->GetBatches<EllpackPage>(&ctx, batch_param)) {
    auto* page = batch.Impl();
    FeatureGroups single_group(page->Cuts());
    DeviceHistogramBuilder builder;
    builder.Reset(&ctx, HistMakerTrainParam::CudaDefaultNodes(),
                  single_group.DeviceAccessor(ctx.Device()), num_categories, false);
    builder.BuildHistogram(ctx.CUDACtx(), page->GetDeviceAccessor(&ctx),
                           single_group.DeviceAccessor(ctx.Device()), gpair.DeviceSpan(), ridx,
                           dh::ToSpan(cat_hist), quantiser);
  }

  /**
   * Generate hist with one hot encoded data.
   */
  auto x_encoded = OneHotEncodeFeature(x, num_categories);
  auto encode_m = GetDMatrixFromData(x_encoded, kRows, num_categories);
  dh::device_vector<GradientPairInt64> encode_hist(2 * num_categories);
  for (auto const &batch : encode_m->GetBatches<EllpackPage>(&ctx, batch_param)) {
    auto* page = batch.Impl();
    FeatureGroups single_group(page->Cuts());
    DeviceHistogramBuilder builder;
    builder.Reset(&ctx, HistMakerTrainParam::CudaDefaultNodes(),
                  single_group.DeviceAccessor(ctx.Device()), encode_hist.size(), false);
    builder.BuildHistogram(ctx.CUDACtx(), page->GetDeviceAccessor(&ctx),
                           single_group.DeviceAccessor(ctx.Device()), gpair.DeviceSpan(), ridx,
                           dh::ToSpan(encode_hist), quantiser);
  }

  std::vector<GradientPairInt64> h_cat_hist(cat_hist.size());
  thrust::copy(cat_hist.begin(), cat_hist.end(), h_cat_hist.begin());

  std::vector<GradientPairInt64> h_encode_hist(encode_hist.size());
  thrust::copy(encode_hist.begin(), encode_hist.end(), h_encode_hist.begin());
  ValidateCategoricalHistogram(num_categories,
                               common::Span<GradientPairInt64>{h_encode_hist},
                               common::Span<GradientPairInt64>{h_cat_hist});
}

TEST(Histogram, GPUHistCategorical) {
  for (size_t num_categories = 2; num_categories < 8; ++num_categories) {
    TestGPUHistogramCategorical(num_categories);
  }
}

namespace {
// Atomic add as type cast for test.
XGBOOST_DEV_INLINE int64_t atomicAdd(int64_t *dst, int64_t src) {  // NOLINT
  uint64_t* u_dst = reinterpret_cast<uint64_t*>(dst);
  uint64_t u_src = *reinterpret_cast<uint64_t*>(&src);
  uint64_t ret = ::atomicAdd(u_dst, u_src);
  return *reinterpret_cast<int64_t*>(&ret);
}
}

void TestAtomicAdd() {
  size_t n_elements = 1024;
  dh::device_vector<int64_t> result_a(1, 0);
  auto d_result_a = result_a.data().get();

  dh::device_vector<int64_t> result_b(1, 0);
  auto d_result_b = result_b.data().get();

  /**
   * Test for simple inputs
   */
  std::vector<int64_t> h_inputs(n_elements);
  for (size_t i = 0; i < h_inputs.size(); ++i) {
    h_inputs[i] = (i % 2 == 0) ? i : -i;
  }
  dh::device_vector<int64_t> inputs(h_inputs);
  auto d_inputs = inputs.data().get();

  dh::LaunchN(n_elements, [=] __device__(size_t i) {
    AtomicAdd64As32(d_result_a, d_inputs[i]);
    atomicAdd(d_result_b, d_inputs[i]);
  });
  ASSERT_EQ(result_a[0], result_b[0]);

  /**
   * Test for positive values that don't fit into 32 bit integer.
   */
  thrust::fill(inputs.begin(), inputs.end(),
               (std::numeric_limits<uint32_t>::max() / 2));
  thrust::fill(result_a.begin(), result_a.end(), 0);
  thrust::fill(result_b.begin(), result_b.end(), 0);
  dh::LaunchN(n_elements, [=] __device__(size_t i) {
    AtomicAdd64As32(d_result_a, d_inputs[i]);
    atomicAdd(d_result_b, d_inputs[i]);
  });
  ASSERT_EQ(result_a[0], result_b[0]);
  ASSERT_GT(result_a[0], std::numeric_limits<uint32_t>::max());
  CHECK_EQ(thrust::reduce(inputs.begin(), inputs.end(), int64_t(0)), result_a[0]);

  /**
   * Test for negative values that don't fit into 32 bit integer.
   */
  thrust::fill(inputs.begin(), inputs.end(),
               (std::numeric_limits<int32_t>::min() / 2));
  thrust::fill(result_a.begin(), result_a.end(), 0);
  thrust::fill(result_b.begin(), result_b.end(), 0);
  dh::LaunchN(n_elements, [=] __device__(size_t i) {
    AtomicAdd64As32(d_result_a, d_inputs[i]);
    atomicAdd(d_result_b, d_inputs[i]);
  });
  ASSERT_EQ(result_a[0], result_b[0]);
  ASSERT_LT(result_a[0], std::numeric_limits<int32_t>::min());
  CHECK_EQ(thrust::reduce(inputs.begin(), inputs.end(), int64_t(0)), result_a[0]);
}

TEST(Histogram, AtomicAddInt64) {
  TestAtomicAdd();
}

TEST(Histogram, Quantiser) {
  auto ctx = MakeCUDACtx(0);
  std::size_t n_samples{16};
  HostDeviceVector<GradientPair> gpair(n_samples, GradientPair{1.0, 1.0});
  gpair.SetDevice(ctx.Device());

  auto quantiser = GradientQuantiser(&ctx, gpair.DeviceSpan(), MetaInfo());
  for (auto v : gpair.ConstHostVector()) {
    auto gh = quantiser.ToFloatingPoint(quantiser.ToFixedPoint(v));
    ASSERT_EQ(gh.GetGrad(), 1.0);
    ASSERT_EQ(gh.GetHess(), 1.0);
  }
}
namespace {
class HistogramExternalMemoryTest : public ::testing::TestWithParam<std::tuple<float, bool>> {
 public:
  void Run(float sparsity, bool force_global) {
    bst_idx_t n_samples{512}, n_features{12}, n_batches{3};
    std::vector<std::unique_ptr<RowPartitioner>> partitioners;
    auto p_fmat = RandomDataGenerator{n_samples, n_features, sparsity}
                      .Batches(n_batches)
                      .GenerateSparsePageDMatrix("cache", true);
    bst_bin_t n_bins = 16;
    BatchParam p{n_bins, TrainParam::DftSparseThreshold()};
    auto ctx = MakeCUDACtx(0);

    std::unique_ptr<FeatureGroups> fg;
    dh::device_vector<GradientPairInt64> single_hist;
    dh::device_vector<GradientPairInt64> multi_hist;

    auto gpair = GenerateRandomGradients(n_samples);
    gpair.SetDevice(ctx.Device());
    auto quantiser = GradientQuantiser{&ctx, gpair.ConstDeviceSpan(), p_fmat->Info()};
    std::shared_ptr<common::HistogramCuts> cuts;

    {
      /**
       * Multi page.
       */
      std::int32_t k{0};
      for (auto const& page : p_fmat->GetBatches<EllpackPage>(&ctx, p)) {
        auto impl = page.Impl();
        if (k == 0) {
          // Initialization
          fg = std::make_unique<FeatureGroups>(impl->Cuts());
          auto init = GradientPairInt64{0, 0};
          multi_hist = decltype(multi_hist)(impl->Cuts().TotalBins(), init);
          single_hist = decltype(single_hist)(impl->Cuts().TotalBins(), init);
          cuts = std::make_shared<common::HistogramCuts>(impl->Cuts());
        }

        partitioners.emplace_back(std::make_unique<RowPartitioner>());
        partitioners.back()->Reset(&ctx, impl->Size(), impl->base_rowid);

        auto ridx = partitioners.at(k)->GetRows(0);
        auto d_histogram = dh::ToSpan(multi_hist);
        DeviceHistogramBuilder builder;
        builder.Reset(&ctx, HistMakerTrainParam::CudaDefaultNodes(),
                      fg->DeviceAccessor(ctx.Device()), d_histogram.size(), force_global);
        builder.BuildHistogram(ctx.CUDACtx(), impl->GetDeviceAccessor(&ctx),
                               fg->DeviceAccessor(ctx.Device()), gpair.ConstDeviceSpan(), ridx,
                               d_histogram, quantiser);
        ++k;
      }
      ASSERT_EQ(k, n_batches);
    }

    {
      /**
       * Single page.
       */
      RowPartitioner partitioner;
      partitioner.Reset(&ctx, p_fmat->Info().num_row_, 0);

      SparsePage concat;
      std::vector<float> hess(p_fmat->Info().num_row_, 1.0f);
      for (auto const& page : p_fmat->GetBatches<SparsePage>()) {
        concat.Push(page);
      }
      EllpackPageImpl page{&ctx, cuts, concat, p_fmat->IsDense(), p_fmat->Info().num_col_, {}};
      auto ridx = partitioner.GetRows(0);
      auto d_histogram = dh::ToSpan(single_hist);
      DeviceHistogramBuilder builder;
      builder.Reset(&ctx, HistMakerTrainParam::CudaDefaultNodes(), fg->DeviceAccessor(ctx.Device()),
                    d_histogram.size(), force_global);
      builder.BuildHistogram(ctx.CUDACtx(), page.GetDeviceAccessor(&ctx),
                             fg->DeviceAccessor(ctx.Device()), gpair.ConstDeviceSpan(), ridx,
                             d_histogram, quantiser);
    }

    std::vector<GradientPairInt64> h_single(single_hist.size());
    thrust::copy(single_hist.begin(), single_hist.end(), h_single.begin());
    std::vector<GradientPairInt64> h_multi(multi_hist.size());
    thrust::copy(multi_hist.begin(), multi_hist.end(), h_multi.begin());

    for (std::size_t i = 0; i < single_hist.size(); ++i) {
      ASSERT_EQ(h_single[i].GetQuantisedGrad(), h_multi[i].GetQuantisedGrad());
      ASSERT_EQ(h_single[i].GetQuantisedHess(), h_multi[i].GetQuantisedHess());
    }
  }
};
}  // namespace

TEST_P(HistogramExternalMemoryTest, ExternalMemory) {
  std::apply(&HistogramExternalMemoryTest::Run, std::tuple_cat(std::make_tuple(this), GetParam()));
}

INSTANTIATE_TEST_SUITE_P(Histogram, HistogramExternalMemoryTest,
                         ::testing::Combine(::testing::Values(0.0f, 0.2f, 0.8f),
                                            ::testing::Bool()));
}  // namespace xgboost::tree
