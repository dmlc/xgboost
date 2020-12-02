#include <gtest/gtest.h>
#include <vector>
#include "../../helpers.h"
#include "../../../../src/common/categorical.h"
#include "../../../../src/tree/gpu_hist/row_partitioner.cuh"
#include "../../../../src/tree/gpu_hist/histogram.cuh"

namespace xgboost {
namespace tree {

template <typename Gradient>
void TestDeterministicHistogram(bool is_dense, int shm_size) {
  size_t constexpr kBins = 256, kCols = 120, kRows = 16384, kRounds = 16;
  float constexpr kLower = -1e-2, kUpper = 1e2;

  float sparsity = is_dense ? 0.0f : 0.5f;
  auto matrix = RandomDataGenerator(kRows, kCols, sparsity).GenerateDMatrix();
  BatchParam batch_param{0, static_cast<int32_t>(kBins), 0};

  for (auto const& batch : matrix->GetBatches<EllpackPage>(batch_param)) {
    auto* page = batch.Impl();

    tree::RowPartitioner row_partitioner(0, kRows);
    auto ridx = row_partitioner.GetRows(0);

    int num_bins = kBins * kCols;
    dh::device_vector<Gradient> histogram(num_bins);
    auto d_histogram = dh::ToSpan(histogram);
    auto gpair = GenerateRandomGradients(kRows, kLower, kUpper);
    gpair.SetDevice(0);

    FeatureGroups feature_groups(page->Cuts(), page->is_dense, shm_size,
                                 sizeof(Gradient));

    auto rounding = CreateRoundingFactor<Gradient>(gpair.DeviceSpan());
    BuildGradientHistogram(page->GetDeviceAccessor(0),
                           feature_groups.DeviceAccessor(0), gpair.DeviceSpan(),
                           ridx, d_histogram, rounding);

    std::vector<Gradient> histogram_h(num_bins);
    dh::safe_cuda(cudaMemcpy(histogram_h.data(), d_histogram.data(),
                             num_bins * sizeof(Gradient),
                             cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < kRounds; ++i) {
      dh::device_vector<Gradient> new_histogram(num_bins);
      auto d_new_histogram = dh::ToSpan(new_histogram);

      auto rounding = CreateRoundingFactor<Gradient>(gpair.DeviceSpan());
      BuildGradientHistogram(page->GetDeviceAccessor(0),
                             feature_groups.DeviceAccessor(0),
                             gpair.DeviceSpan(), ridx, d_new_histogram,
                             rounding);

      std::vector<Gradient> new_histogram_h(num_bins);
      dh::safe_cuda(cudaMemcpy(new_histogram_h.data(), d_new_histogram.data(),
                               num_bins * sizeof(Gradient),
                               cudaMemcpyDeviceToHost));
      for (size_t j = 0; j < new_histogram_h.size(); ++j) {
        ASSERT_EQ(new_histogram_h[j].GetGrad(), histogram_h[j].GetGrad());
        ASSERT_EQ(new_histogram_h[j].GetHess(), histogram_h[j].GetHess());
      }
    }

    {
      auto gpair = GenerateRandomGradients(kRows, kLower, kUpper);
      gpair.SetDevice(0);

      // Use a single feature group to compute the baseline.
      FeatureGroups single_group(page->Cuts());

      dh::device_vector<Gradient> baseline(num_bins);
      BuildGradientHistogram(page->GetDeviceAccessor(0),
                             single_group.DeviceAccessor(0),
                             gpair.DeviceSpan(), ridx, dh::ToSpan(baseline),
                             rounding);

      std::vector<Gradient> baseline_h(num_bins);
      dh::safe_cuda(cudaMemcpy(baseline_h.data(), baseline.data().get(),
                               num_bins * sizeof(Gradient),
                               cudaMemcpyDeviceToHost));

      for (size_t i = 0; i < baseline.size(); ++i) {
        EXPECT_NEAR(baseline_h[i].GetGrad(), histogram_h[i].GetGrad(),
                    baseline_h[i].GetGrad() * 1e-3);
      }
    }
  }
}

TEST(Histogram, GPUDeterministic) {
  std::vector<bool> is_dense_array{false, true};
  std::vector<int> shm_sizes{48 * 1024, 64 * 1024, 160 * 1024};
  for (bool is_dense : is_dense_array) {
    for (int shm_size : shm_sizes) {
      TestDeterministicHistogram<GradientPair>(is_dense, shm_size);
      TestDeterministicHistogram<GradientPairPrecise>(is_dense, shm_size);
    }
  }
}

std::vector<float> OneHotEncodeFeature(std::vector<float> x, size_t num_cat) {
  std::vector<float> ret(x.size() * num_cat, 0);
  size_t n_rows = x.size();
  for (size_t r = 0; r < n_rows; ++r) {
    bst_cat_t cat = common::AsCat(x[r]);
    ret.at(num_cat * r + cat) = 1;
  }
  return ret;
}

// Test 1 vs rest categorical histogram is equivalent to one hot encoded data.
void TestGPUHistogramCategorical(size_t num_categories) {
  size_t constexpr kRows = 340;
  size_t constexpr kBins = 256;
  auto x = GenerateRandomCategoricalSingleColumn(kRows, num_categories);
  auto cat_m = GetDMatrixFromData(x, kRows, 1);
  cat_m->Info().feature_types.HostVector().push_back(FeatureType::kCategorical);
  BatchParam batch_param{0, static_cast<int32_t>(kBins), 0};
  tree::RowPartitioner row_partitioner(0, kRows);
  auto ridx = row_partitioner.GetRows(0);
  dh::device_vector<GradientPairPrecise> cat_hist(num_categories);
  auto gpair = GenerateRandomGradients(kRows, 0, 2);
  gpair.SetDevice(0);
  auto rounding = CreateRoundingFactor<GradientPairPrecise>(gpair.DeviceSpan());
  // Generate hist with cat data.
  for (auto const &batch : cat_m->GetBatches<EllpackPage>(batch_param)) {
    auto* page = batch.Impl();
    FeatureGroups single_group(page->Cuts());
    BuildGradientHistogram(page->GetDeviceAccessor(0),
                           single_group.DeviceAccessor(0),
                           gpair.DeviceSpan(), ridx, dh::ToSpan(cat_hist),
                           rounding);
  }

  // Generate hist with one hot encoded data.
  auto x_encoded = OneHotEncodeFeature(x, num_categories);
  auto encode_m = GetDMatrixFromData(x_encoded, kRows, num_categories);
  dh::device_vector<GradientPairPrecise> encode_hist(2 * num_categories);
  for (auto const &batch : encode_m->GetBatches<EllpackPage>(batch_param)) {
    auto* page = batch.Impl();
    FeatureGroups single_group(page->Cuts());
    BuildGradientHistogram(page->GetDeviceAccessor(0),
                           single_group.DeviceAccessor(0),
                           gpair.DeviceSpan(), ridx, dh::ToSpan(encode_hist),
                           rounding);
  }

  std::vector<GradientPairPrecise> h_cat_hist(cat_hist.size());
  thrust::copy(cat_hist.begin(), cat_hist.end(), h_cat_hist.begin());
  auto cat_sum = std::accumulate(h_cat_hist.begin(), h_cat_hist.end(), GradientPairPrecise{});

  std::vector<GradientPairPrecise> h_encode_hist(encode_hist.size());
  thrust::copy(encode_hist.begin(), encode_hist.end(), h_encode_hist.begin());

  for (size_t c = 0; c < num_categories; ++c) {
    auto zero = h_encode_hist[c * 2];
    auto one = h_encode_hist[c * 2 + 1];

    auto chosen = h_cat_hist[c];
    auto not_chosen = cat_sum - chosen;

    ASSERT_LE(RelError(zero.GetGrad(), not_chosen.GetGrad()), kRtEps);
    ASSERT_LE(RelError(zero.GetHess(), not_chosen.GetHess()), kRtEps);

    ASSERT_LE(RelError(one.GetGrad(), chosen.GetGrad()), kRtEps);
    ASSERT_LE(RelError(one.GetHess(), chosen.GetHess()), kRtEps);
  }
}

TEST(Histogram, GPUHistCategorical) {
  for (size_t num_categories = 2; num_categories < 8; ++num_categories) {
    TestGPUHistogramCategorical(num_categories);
  }
}
}  // namespace tree
}  // namespace xgboost
