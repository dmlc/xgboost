/**
 * Copyright 2020-2023, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <vector>

#include "../../../../src/common/categorical.h"
#include "../../../../src/tree/gpu_hist/histogram.cuh"
#include "../../../../src/tree/gpu_hist/row_partitioner.cuh"
#include "../../../../src/tree/param.h"  // TrainParam
#include "../../categorical_helpers.h"
#include "../../helpers.h"

namespace xgboost {
namespace tree {

void TestDeterministicHistogram(bool is_dense, int shm_size) {
  Context ctx = MakeCUDACtx(0);
  size_t constexpr kBins = 256, kCols = 120, kRows = 16384, kRounds = 16;
  float constexpr kLower = -1e-2, kUpper = 1e2;

  float sparsity = is_dense ? 0.0f : 0.5f;
  auto matrix = RandomDataGenerator(kRows, kCols, sparsity).GenerateDMatrix();
  auto batch_param = BatchParam{kBins, tree::TrainParam::DftSparseThreshold()};

  for (auto const& batch : matrix->GetBatches<EllpackPage>(&ctx, batch_param)) {
    auto* page = batch.Impl();

    tree::RowPartitioner row_partitioner(0, kRows);
    auto ridx = row_partitioner.GetRows(0);

    int num_bins = kBins * kCols;
    dh::device_vector<GradientPairInt64> histogram(num_bins);
    auto d_histogram = dh::ToSpan(histogram);
    auto gpair = GenerateRandomGradients(kRows, kLower, kUpper);
    gpair.SetDevice(0);

    FeatureGroups feature_groups(page->Cuts(), page->is_dense, shm_size,
                                 sizeof(GradientPairInt64));

    auto quantiser = GradientQuantiser(gpair.DeviceSpan());
    BuildGradientHistogram(ctx.CUDACtx(), page->GetDeviceAccessor(0),
                           feature_groups.DeviceAccessor(0), gpair.DeviceSpan(), ridx, d_histogram,
                           quantiser);

    std::vector<GradientPairInt64> histogram_h(num_bins);
    dh::safe_cuda(cudaMemcpy(histogram_h.data(), d_histogram.data(),
                             num_bins * sizeof(GradientPairInt64),
                             cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < kRounds; ++i) {
      dh::device_vector<GradientPairInt64> new_histogram(num_bins);
      auto d_new_histogram = dh::ToSpan(new_histogram);

      auto quantiser = GradientQuantiser(gpair.DeviceSpan());
      BuildGradientHistogram(ctx.CUDACtx(), page->GetDeviceAccessor(0),
                             feature_groups.DeviceAccessor(0), gpair.DeviceSpan(), ridx,
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
      gpair.SetDevice(0);

      // Use a single feature group to compute the baseline.
      FeatureGroups single_group(page->Cuts());

      dh::device_vector<GradientPairInt64> baseline(num_bins);
      BuildGradientHistogram(ctx.CUDACtx(), page->GetDeviceAccessor(0),
                             single_group.DeviceAccessor(0), gpair.DeviceSpan(), ridx,
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
      TestDeterministicHistogram(is_dense, shm_size);
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
  tree::RowPartitioner row_partitioner(0, kRows);
  auto ridx = row_partitioner.GetRows(0);
  dh::device_vector<GradientPairInt64> cat_hist(num_categories);
  auto gpair = GenerateRandomGradients(kRows, 0, 2);
  gpair.SetDevice(0);
  auto quantiser = GradientQuantiser(gpair.DeviceSpan());
  /**
   * Generate hist with cat data.
   */
  for (auto const &batch : cat_m->GetBatches<EllpackPage>(&ctx, batch_param)) {
    auto* page = batch.Impl();
    FeatureGroups single_group(page->Cuts());
    BuildGradientHistogram(ctx.CUDACtx(), page->GetDeviceAccessor(0),
                           single_group.DeviceAccessor(0), gpair.DeviceSpan(), ridx,
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
    BuildGradientHistogram(ctx.CUDACtx(), page->GetDeviceAccessor(0),
                           single_group.DeviceAccessor(0), gpair.DeviceSpan(), ridx,
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
}  // namespace tree
}  // namespace xgboost
