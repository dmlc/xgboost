#include <gtest/gtest.h>
#include <vector>
#include <thrust/shuffle.h>
#include <thrust/random.h>

#include "../../../../src/common/categorical.h"
#include "../../../../src/tree/gpu_hist/histogram.cuh"
#include "../../../../src/tree/gpu_hist/row_partitioner.cuh"
#include "../../categorical_helpers.h"
#include "../../helpers.h"

namespace xgboost {
namespace tree {

template <typename Gradient>
void TestDeterministicHistogram(bool is_dense, int shm_size) {
  size_t constexpr kBins = 256, kCols = 120, kRows = 16384, kRounds = 16;
  float constexpr kLower = -1e-2, kUpper = 1e2;

  float sparsity = is_dense ? 0.0f : 0.5f;
  auto matrix = RandomDataGenerator(kRows, kCols, sparsity).GenerateDMatrix();
  BatchParam batch_param{0, static_cast<int32_t>(kBins)};

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
      TestDeterministicHistogram<GradientPairPrecise>(is_dense, shm_size);
    }
  }
}

// Test 1 vs rest categorical histogram is equivalent to one hot encoded data.
void TestGPUHistogramCategorical(size_t num_categories) {
  size_t constexpr kRows = 340;
  size_t constexpr kBins = 256;
  auto x = GenerateRandomCategoricalSingleColumn(kRows, num_categories);
  auto cat_m = GetDMatrixFromData(x, kRows, 1);
  cat_m->Info().feature_types.HostVector().push_back(FeatureType::kCategorical);
  BatchParam batch_param{0, static_cast<int32_t>(kBins)};
  tree::RowPartitioner row_partitioner(0, kRows);
  auto ridx = row_partitioner.GetRows(0);
  dh::device_vector<GradientPairPrecise> cat_hist(num_categories);
  auto gpair = GenerateRandomGradients(kRows, 0, 2);
  gpair.SetDevice(0);
  auto rounding = CreateRoundingFactor<GradientPairPrecise>(gpair.DeviceSpan());
  /**
   * Generate hist with cat data.
   */
  for (auto const &batch : cat_m->GetBatches<EllpackPage>(batch_param)) {
    auto* page = batch.Impl();
    FeatureGroups single_group(page->Cuts());
    BuildGradientHistogram(page->GetDeviceAccessor(0),
                           single_group.DeviceAccessor(0),
                           gpair.DeviceSpan(), ridx, dh::ToSpan(cat_hist),
                           rounding);
  }

  /**
   * Generate hist with one hot encoded data.
   */
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

  std::vector<GradientPairPrecise> h_encode_hist(encode_hist.size());
  thrust::copy(encode_hist.begin(), encode_hist.end(), h_encode_hist.begin());
  ValidateCategoricalHistogram(num_categories,
                               common::Span<GradientPairPrecise>{h_encode_hist},
                               common::Span<GradientPairPrecise>{h_cat_hist});
}

TEST(Histogram, GPUHistCategorical) {
  for (size_t num_categories = 2; num_categories < 8; ++num_categories) {
    TestGPUHistogramCategorical(num_categories);
  }
}

void RunBenchmark(std::string branch, std::string name, std::size_t kCols, std::size_t kRows, float sparsity, int max_depth){
  size_t constexpr kBins = 256;
  float constexpr kLower = -1e-2, kUpper = 1e2;
  auto matrix = RandomDataGenerator(kRows, kCols, sparsity).GenerateDMatrix();
  BatchParam batch_param{0, static_cast<int32_t>(kBins)};
  int num_bins = kBins * kCols;
  dh::device_vector<GradientPairPrecise> histogram(num_bins);
  auto d_histogram = dh::ToSpan(histogram);
  auto gpair = GenerateRandomGradients(kRows, kLower, kUpper);
  gpair.SetDevice(0);
  auto rounding = CreateRoundingFactor<GradientPairPrecise>(gpair.DeviceSpan());
  thrust::device_vector<uint32_t> ridx(kRows);
  for (auto const& batch : matrix->GetBatches<EllpackPage>(batch_param)) {
    auto* page = batch.Impl();
    FeatureGroups feature_groups(page->Cuts(), page->is_dense, dh::MaxSharedMemoryOptin(0),
                                 sizeof(GradientPairPrecise));
    for (int depth = 1; depth <= max_depth; depth++) {
      std::size_t depth_rows = kRows / (1 << (depth - 1));
      std::cout << branch << ',' << name << ',' << depth << ',' << depth_rows << ',' << kCols
                << ',';
      thrust::shuffle(ridx.begin(), ridx.end(), thrust::default_random_engine(depth));
      auto d_ridx = dh::ToSpan(ridx).subspan(0, depth_rows);
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      cudaEventRecord(start);
      BuildGradientHistogram(page->GetDeviceAccessor(0), feature_groups.DeviceAccessor(0),
                             gpair.DeviceSpan(), d_ridx, d_histogram, rounding);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);

      std::cout << milliseconds << ',' << ((depth_rows * kCols) / milliseconds)/1000 << '\n';
    }
  }
}

TEST(Histogram, Benchmark) {
  float sparsity = 0.0f;
  std::string branch("overflow");
  std::cout << "branch,dataset,depth,rows,cols,time(ms),Million elements/s\n";
  int max_depth = 8;
  RunBenchmark(branch, "epsilon", 2000, 500000, sparsity, max_depth);
  RunBenchmark(branch, "higgs", 32, 10000000, sparsity, max_depth);
  RunBenchmark(branch, "airline", 13, 115000000, sparsity, max_depth);
  RunBenchmark(branch, "year", 90, 515345, sparsity, max_depth);
}


void TestAtomicAddWithOverflow() {
  thrust::device_vector<GradientPairPrecise> histogram(2);
  thrust::device_vector<GradientPair> gpair = std::vector<GradientPair>{{1.0, 1.0}, {-0.01, 0.1}, {0.02, 0.1}, {-2.0, 1.0}};
  auto d_gpair = dh::ToSpan(gpair);
  auto rounding = CreateRoundingFactor<GradientPairPrecise>(d_gpair);
  auto d_histogram = histogram.data().get();
  dh::LaunchN(gpair.size(), [=] __device__(int idx) {
    __shared__ char shared[sizeof(GradientPairInt32)];
    auto shared_histogram = reinterpret_cast<GradientPairInt32*>(shared);
    if (idx == 0) {
      shared_histogram[0] = GradientPairInt32();
    }

    // Global memory version
    GradientPairPrecise truncated{
        TruncateWithRoundingFactor<GradientPairPrecise::ValueT>(rounding.rounding.GetGrad(),
                                                         d_gpair[idx].GetGrad()),
        TruncateWithRoundingFactor<GradientPairPrecise::ValueT>(rounding.rounding.GetHess(),
                                                         d_gpair[idx].GetHess()),
    };
    dh::AtomicAddGpair(d_histogram, truncated);

    // Reduced precision shared memory version
    auto adjusted = rounding.ToFixedPoint(d_gpair[idx]);
    AtomicAddGpairWithOverflow(shared_histogram, adjusted, d_histogram + 1, rounding);
    // First thread copies shared back to global
    if (idx == 0) {
      dh::AtomicAddGpair(d_histogram + 1, rounding.ToFloatingPoint(GradientPairInt64{shared_histogram[idx].GetGrad(),shared_histogram[idx].GetHess()}));
    }
  });

  GradientPairPrecise global = histogram[0];
  GradientPairPrecise shared = histogram[1];
  ASSERT_EQ(global.GetGrad(), shared.GetGrad());
  ASSERT_EQ(global.GetHess(), shared.GetHess());
}

TEST(Histogram, AtomicAddWithOverflow) {
TestAtomicAddWithOverflow();
}

}  // namespace tree
}  // namespace xgboost
