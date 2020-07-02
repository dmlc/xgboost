#include <gtest/gtest.h>
#include <vector>
#include "../../helpers.h"
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

    FeatureGroups feature_groups;
    feature_groups.Init<Gradient>(page->Cuts(), page->is_dense, shm_size);
    
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
      FeatureGroups single_group;
      single_group.InitSingle(page->Cuts());
      
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
}  // namespace tree
}  // namespace xgboost
