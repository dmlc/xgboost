/**
 * Copyright 2020-2024, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>  // for Context

#include <memory>  // for unique_ptr
#include <vector>  // for vector

#include "../../../../src/tree/gpu_hist/histogram.cuh"
#include "../../../../src/tree/gpu_hist/row_partitioner.cuh"  // for RowPartitioner
#include "../../../../src/tree/param.h"                       // for TrainParam
#include "../../categorical_helpers.h"                        // for OneHotEncodeFeature
#include "../../helpers.h"

namespace xgboost::tree {
void TestDeterministicHistogram(bool is_dense, int shm_size, bool force_global) {
  Context ctx = MakeCUDACtx(0);
  size_t constexpr kBins = 256, kCols = 120, kRows = 16384, kRounds = 16;
  float constexpr kLower = -1e-2, kUpper = 1e2;

  float sparsity = is_dense ? 0.0f : 0.5f;
  auto matrix = RandomDataGenerator(kRows, kCols, sparsity).GenerateDMatrix();
  auto batch_param = BatchParam{kBins, tree::TrainParam::DftSparseThreshold()};

  for (auto const& batch : matrix->GetBatches<EllpackPage>(&ctx, batch_param)) {
    auto* page = batch.Impl();

    tree::RowPartitioner row_partitioner{&ctx, kRows, page->base_rowid};
    auto ridx = row_partitioner.GetRows(0);

    bst_bin_t num_bins = kBins * kCols;
    dh::device_vector<GradientPairInt64> histogram(num_bins);
    auto d_histogram = dh::ToSpan(histogram);
    auto gpair = GenerateRandomGradients(kRows, kLower, kUpper);
    gpair.SetDevice(ctx.Device());

    FeatureGroups feature_groups(page->Cuts(), page->is_dense, shm_size, sizeof(GradientPairInt64));

    auto quantiser = GradientQuantiser(&ctx, gpair.DeviceSpan(), MetaInfo());
    DeviceHistogramBuilder builder;
    builder.Reset(&ctx, feature_groups.DeviceAccessor(ctx.Device()), force_global);
    builder.BuildHistogram(ctx.CUDACtx(), page->GetDeviceAccessor(ctx.Device()),
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
      builder.Reset(&ctx, feature_groups.DeviceAccessor(ctx.Device()), force_global);
      builder.BuildHistogram(ctx.CUDACtx(), page->GetDeviceAccessor(ctx.Device()),
                             feature_groups.DeviceAccessor(ctx.Device()), gpair.DeviceSpan(), ridx,
                             d_new_histogram, quantiser);

      std::vector<GradientPairInt64> new_histogram_h(num_bins);
      dh::safe_cuda(cudaMemcpy(new_histogram_h.data(), d_new_histogram.data(),
                               num_bins * sizeof(GradientPairInt64),
                               cudaMemcpyDeviceToHost));
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
      builder.Reset(&ctx, single_group.DeviceAccessor(ctx.Device()), force_global);
      builder.BuildHistogram(ctx.CUDACtx(), page->GetDeviceAccessor(ctx.Device()),
                             single_group.DeviceAccessor(ctx.Device()), gpair.DeviceSpan(), ridx,
                             dh::ToSpan(baseline), quantiser);

      std::vector<GradientPairInt64> baseline_h(num_bins);
      dh::safe_cuda(cudaMemcpy(baseline_h.data(), baseline.data().get(),
                               num_bins * sizeof(GradientPairInt64),
                               cudaMemcpyDeviceToHost));

      for (size_t i = 0; i < baseline.size(); ++i) {
        EXPECT_NEAR(baseline_h[i].GetQuantisedGrad(), histogram_h[i].GetQuantisedGrad(),
                    baseline_h[i].GetQuantisedGrad() * 1e-3);
      }
    }
  }
}

TEST(Histogram, GPUDeterministic) {
  std::vector<bool> is_dense_array{false, true};
  std::vector<int> shm_sizes{48 * 1024, 64 * 1024, 160 * 1024};
  for (bool is_dense : is_dense_array) {
    for (int shm_size : shm_sizes) {
      for (bool force_global : {true, false}) {
        TestDeterministicHistogram(is_dense, shm_size, force_global);
      }
    }
  }
}

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
  tree::RowPartitioner row_partitioner{&ctx, kRows, 0};
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
    builder.Reset(&ctx, single_group.DeviceAccessor(ctx.Device()), false);
    builder.BuildHistogram(ctx.CUDACtx(), page->GetDeviceAccessor(ctx.Device()),
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
    builder.Reset(&ctx, single_group.DeviceAccessor(ctx.Device()), false);
    builder.BuildHistogram(ctx.CUDACtx(), page->GetDeviceAccessor(ctx.Device()),
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
          auto d_matrix = impl->GetDeviceAccessor(ctx.Device());
          fg = std::make_unique<FeatureGroups>(impl->Cuts());
          auto init = GradientPairInt64{0, 0};
          multi_hist = decltype(multi_hist)(impl->Cuts().TotalBins(), init);
          single_hist = decltype(single_hist)(impl->Cuts().TotalBins(), init);
          cuts = std::make_shared<common::HistogramCuts>(impl->Cuts());
        }

        partitioners.emplace_back(
            std::make_unique<RowPartitioner>(&ctx, impl->Size(), impl->base_rowid));

        auto ridx = partitioners.at(k)->GetRows(0);
        auto d_histogram = dh::ToSpan(multi_hist);
        DeviceHistogramBuilder builder;
        builder.Reset(&ctx, fg->DeviceAccessor(ctx.Device()), force_global);
        builder.BuildHistogram(ctx.CUDACtx(), impl->GetDeviceAccessor(ctx.Device()),
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
      RowPartitioner partitioner{&ctx, p_fmat->Info().num_row_, 0};
      SparsePage concat;
      std::vector<float> hess(p_fmat->Info().num_row_, 1.0f);
      for (auto const& page : p_fmat->GetBatches<SparsePage>()) {
        concat.Push(page);
      }
      EllpackPageImpl page{
          ctx.Device(), cuts, concat, p_fmat->IsDense(), p_fmat->Info().num_col_, {}};
      auto ridx = partitioner.GetRows(0);
      auto d_histogram = dh::ToSpan(single_hist);
      DeviceHistogramBuilder builder;
      builder.Reset(&ctx, fg->DeviceAccessor(ctx.Device()), force_global);
      builder.BuildHistogram(ctx.CUDACtx(), page.GetDeviceAccessor(ctx.Device()),
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

INSTANTIATE_TEST_SUITE_P(Histogram, HistogramExternalMemoryTest, ::testing::ValuesIn([]() {
                           std::vector<std::tuple<float, bool>> params;
                           for (auto global : {true, false}) {
                             for (auto sparsity : {0.0f, 0.2f, 0.8f}) {
                               params.emplace_back(sparsity, global);
                             }
                           }
                           return params;
                         }()));
}  // namespace xgboost::tree
