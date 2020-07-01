#include <gtest/gtest.h>
#include "../../helpers.h"
#include "../../../../src/tree/gpu_hist/row_partitioner.cuh"
#include "../../../../src/tree/gpu_hist/histogram.cuh"

namespace xgboost {
namespace tree {

template <typename Gradient>
void TestDeterminsticHistogram(bool is_dense, int shm_size) {
  size_t constexpr kBins = 24, kCols = 8, kRows = 32768, kRounds = 16;
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

    for (size_t i = 0; i < kRounds; ++i) {
      dh::device_vector<Gradient> new_histogram(kBins * kCols);
      auto d_histogram = dh::ToSpan(new_histogram);

      auto rounding = CreateRoundingFactor<Gradient>(gpair.DeviceSpan());
      BuildGradientHistogram(page->GetDeviceAccessor(0),
                             feature_groups.DeviceAccessor(0),
                             gpair.DeviceSpan(), ridx,  d_histogram, rounding);

      for (size_t j = 0; j < new_histogram.size(); ++j) {
        ASSERT_EQ(((Gradient)new_histogram[j]).GetGrad(),
                  ((Gradient)histogram[j]).GetGrad());
        ASSERT_EQ(((Gradient)new_histogram[j]).GetHess(),
                  ((Gradient)histogram[j]).GetHess());
      }
    }

    {
      auto gpair = GenerateRandomGradients(kRows, kLower, kUpper);
      gpair.SetDevice(0);
      dh::device_vector<Gradient> baseline(kBins * kCols);
      BuildGradientHistogram(page->GetDeviceAccessor(0),
                             feature_groups.DeviceAccessor(0),
                             gpair.DeviceSpan(), ridx, dh::ToSpan(baseline),
                             rounding);
      for (size_t i = 0; i < baseline.size(); ++i) {
        EXPECT_NEAR(((Gradient)baseline[i]).GetGrad(), ((Gradient)histogram[i]).GetGrad(),
                    ((Gradient)baseline[i]).GetGrad() * 1e-3);
      }
    }
  }
}

TEST(Histogram, GPUDeterminstic) {
  std::vector<bool> is_dense_array{false, true};
  std::vector<int> shm_sizes{48 * 1024, 64 * 1024, 96 * 1024};
  for (bool is_dense : is_dense_array) {
    for (int shm_size : shm_sizes) {
      TestDeterminsticHistogram<GradientPair>(is_dense, shm_size);
      TestDeterminsticHistogram<GradientPairPrecise>(is_dense, shm_size);
    }
  }
}
}  // namespace tree
}  // namespace xgboost
