/**
 * Copyright 2019-2024, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <xgboost/base.h>  // for bst_bin_t
#include <xgboost/c_api.h>
#include <xgboost/data.h>

#include <algorithm>  // for transform
#include <cmath>      // for floor
#include <cstddef>    // for size_t
#include <limits>     // for numeric_limits
#include <string>     // for string, to_string
#include <tuple>      // for tuple, make_tuple
#include <vector>     // for vector

#include "../../../include/xgboost/logging.h"
#include "../../../src/common/cuda_context.cuh"
#include "../../../src/common/cuda_rt_utils.h"  // for SetDevice
#include "../../../src/common/device_helpers.cuh"
#include "../../../src/common/hist_util.cuh"
#include "../../../src/common/hist_util.h"
#include "../../../src/data/device_adapter.cuh"
#include "../../../src/data/simple_dmatrix.h"
#include "../data/test_array_interface.h"
#include "../filesystem.h"  // dmlc::TemporaryDirectory
#include "../helpers.h"
#include "test_hist_util.h"

namespace xgboost::common {

template <typename AdapterT>
HistogramCuts GetHostCuts(Context const* ctx, AdapterT* adapter, int num_bins, float missing) {
  data::SimpleDMatrix dmat(adapter, missing, 1);
  HistogramCuts cuts = SketchOnDMatrix(ctx, &dmat, num_bins);
  return cuts;
}

TEST(HistUtil, DeviceSketch) {
  auto ctx = MakeCUDACtx(0);
  int num_columns = 1;
  int num_bins = 4;
  std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 7.0f, -1.0f};
  int num_rows = x.size();
  auto dmat = GetDMatrixFromData(x, num_rows, num_columns);

  auto device_cuts = DeviceSketch(&ctx, dmat.get(), num_bins);

  Context cpu_ctx;
  HistogramCuts host_cuts = SketchOnDMatrix(&cpu_ctx, dmat.get(), num_bins);

  EXPECT_EQ(device_cuts.Values(), host_cuts.Values());
  EXPECT_EQ(device_cuts.Ptrs(), host_cuts.Ptrs());
  EXPECT_EQ(device_cuts.MinValues(), host_cuts.MinValues());
}

TEST(HistUtil, SketchBatchNumElements) {
#if defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1
  GTEST_SKIP_("Test not runnable with RMM enabled.");
#endif  // defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1
  size_t constexpr kCols = 10000;
  std::int32_t device = dh::CurrentDevice();
  auto avail = static_cast<size_t>(dh::AvailableMemory(device) * 0.8);
  auto per_elem = detail::BytesPerElement(false);
  auto avail_elem = avail / per_elem;
  size_t rows = avail_elem / kCols * 10;
  auto shape = detail::SketchShape{rows, kCols, rows * kCols};
  auto batch = detail::SketchBatchNumElements(detail::UnknownSketchNumElements(), shape, device,
                                              256, false, 0);
  ASSERT_EQ(batch, avail_elem);
}

TEST(HistUtil, DeviceSketchMemory) {
  auto ctx = MakeCUDACtx(0);
  int num_columns = 100;
  int num_rows = 1000;
  int num_bins = 256;
  auto x = GenerateRandom(num_rows, num_columns);
  auto dmat = GetDMatrixFromData(x, num_rows, num_columns);

  dh::GlobalMemoryLogger().Clear();
  ConsoleLogger::Configure({{"verbosity", "3"}});
  auto device_cuts = DeviceSketch(&ctx, dmat.get(), num_bins);

  size_t bytes_required = detail::RequiredMemory(
      num_rows, num_columns, num_rows * num_columns, num_bins, false);
  EXPECT_LE(dh::GlobalMemoryLogger().PeakMemory(), bytes_required * 1.05);
  EXPECT_GE(dh::GlobalMemoryLogger().PeakMemory(), bytes_required * 0.95);
  ConsoleLogger::Configure({{"verbosity", "0"}});
}

TEST(HistUtil, DeviceSketchWeightsMemory) {
  auto ctx = MakeCUDACtx(0);
  int num_columns = 100;
  int num_rows = 1000;
  int num_bins = 256;
  auto x = GenerateRandom(num_rows, num_columns);
  auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
  dmat->Info().weights_.HostVector() = GenerateRandomWeights(num_rows);

  dh::GlobalMemoryLogger().Clear();
  ConsoleLogger::Configure({{"verbosity", "3"}});
  auto device_cuts = DeviceSketch(&ctx, dmat.get(), num_bins);
  ConsoleLogger::Configure({{"verbosity", "0"}});

  size_t bytes_required = detail::RequiredMemory(
      num_rows, num_columns, num_rows * num_columns, num_bins, true);
  EXPECT_LE(dh::GlobalMemoryLogger().PeakMemory(), bytes_required * 1.05);
  EXPECT_GE(dh::GlobalMemoryLogger().PeakMemory(), bytes_required);
}

TEST(HistUtil, DeviceSketchDeterminism) {
  auto ctx = MakeCUDACtx(0);
  int num_rows = 500;
  int num_columns = 5;
  int num_bins = 256;
  auto x = GenerateRandom(num_rows, num_columns);
  auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
  auto reference_sketch = DeviceSketch(&ctx, dmat.get(), num_bins);
  size_t constexpr kRounds{ 100 };
  for (size_t r = 0; r < kRounds; ++r) {
    auto new_sketch = DeviceSketch(&ctx, dmat.get(), num_bins);
    ASSERT_EQ(reference_sketch.Values(), new_sketch.Values());
    ASSERT_EQ(reference_sketch.MinValues(), new_sketch.MinValues());
  }
}

TEST(HistUtil, DeviceSketchCategoricalAsNumeric) {
  auto ctx = MakeCUDACtx(0);
  auto categorical_sizes = {2, 6, 8, 12};
  int num_bins = 256;
  auto sizes = {25, 100, 1000};
  for (auto n : sizes) {
    for (auto num_categories : categorical_sizes) {
      auto x = GenerateRandomCategoricalSingleColumn(n, num_categories);
      auto dmat = GetDMatrixFromData(x, n, 1);
      auto cuts = DeviceSketch(&ctx, dmat.get(), num_bins);
      ValidateCuts(cuts, dmat.get(), num_bins);
    }
  }
}

TEST(HistUtil, DeviceSketchCategoricalFeatures) {
  auto ctx = MakeCUDACtx(0);
  TestCategoricalSketch(1000, 256, 32, false, [ctx](DMatrix* p_fmat, int32_t num_bins) {
    return DeviceSketch(&ctx, p_fmat, num_bins);
  });
  TestCategoricalSketch(1000, 256, 32, true, [ctx](DMatrix* p_fmat, int32_t num_bins) {
    return DeviceSketch(&ctx, p_fmat, num_bins);
  });
}

void TestMixedSketch() {
  size_t n_samples = 1000, n_features = 2, n_categories = 3;
  bst_bin_t n_bins = 64;

  std::vector<float> data(n_samples * n_features);
  SimpleLCG gen;
  SimpleRealUniformDistribution<float> cat_d{0.0f, static_cast<float>(n_categories)};
  SimpleRealUniformDistribution<float> num_d{0.0f, 3.0f};
  for (size_t i = 0; i < n_samples * n_features; ++i) {
    // two features, row major. The first column is numeric and the second is categorical.
    if (i % 2 == 0) {
      data[i] = std::floor(cat_d(&gen));
    } else {
      data[i] = num_d(&gen);
    }
  }

  auto m = GetDMatrixFromData(data, n_samples, n_features);
  m->Info().feature_types.HostVector().push_back(FeatureType::kCategorical);
  m->Info().feature_types.HostVector().push_back(FeatureType::kNumerical);

  auto ctx = MakeCUDACtx(0);
  auto cuts = DeviceSketch(&ctx, m.get(), n_bins);
  ASSERT_EQ(cuts.Values().size(), n_bins + n_categories);
}

TEST(HistUtil, DeviceSketchMixedFeatures) { TestMixedSketch(); }

TEST(HistUtil, RemoveDuplicatedCategories) {
  bst_idx_t n_samples = 512;
  bst_feature_t n_features = 3;
  bst_cat_t n_categories = 5;

  auto ctx = MakeCUDACtx(0);
  SimpleLCG rng;
  SimpleRealUniformDistribution<float> cat_d{0.0f, static_cast<float>(n_categories)};

  dh::device_vector<Entry> sorted_entries(n_samples * n_features);
  for (std::size_t i = 0; i < n_samples; ++i) {
    for (bst_feature_t j = 0; j < n_features; ++j) {
      float fvalue{0.0f};
      // The second column is categorical
      if (j == 1) {
        fvalue = std::floor(cat_d(&rng));
      } else {
        fvalue = i;
      }
      sorted_entries[i * n_features + j] = Entry{j, fvalue};
    }
  }

  MetaInfo info;
  info.num_col_ = n_features;
  info.num_row_ = n_samples;
  info.feature_types.HostVector() = std::vector<FeatureType>{
      FeatureType::kNumerical, FeatureType::kCategorical, FeatureType::kNumerical};
  ASSERT_EQ(info.feature_types.Size(), n_features);

  HostDeviceVector<bst_idx_t> cuts_ptr{0, n_samples, n_samples * 2, n_samples * 3};
  cuts_ptr.SetDevice(DeviceOrd::CUDA(0));

  dh::device_vector<float> weight(n_samples * n_features, 0);
  dh::Iota(dh::ToSpan(weight), ctx.CUDACtx()->Stream());

  dh::caching_device_vector<bst_idx_t> columns_ptr(4);
  for (std::size_t i = 0; i < columns_ptr.size(); ++i) {
    columns_ptr[i] = i * n_samples;
  }
  // sort into column major
  thrust::sort_by_key(sorted_entries.begin(), sorted_entries.end(), weight.begin(),
                      detail::EntryCompareOp());

  detail::RemoveDuplicatedCategories(&ctx, info, cuts_ptr.DeviceSpan(), &sorted_entries, &weight,
                                     &columns_ptr);

  auto const& h_cptr = cuts_ptr.ConstHostVector();
  ASSERT_EQ(h_cptr.back(), n_samples * 2 + n_categories);
  // check numerical
  for (std::size_t i = 0; i < n_samples; ++i) {
    ASSERT_EQ(weight[i], i * 3);
  }
  auto beg = n_samples + n_categories;
  for (std::size_t i = 0; i < n_samples; ++i) {
    ASSERT_EQ(weight[i + beg], i * 3 + 2);
  }
  // check categorical
  beg = n_samples;
  for (bst_cat_t i = 0; i < n_categories; ++i) {
    // all from the second column
    ASSERT_EQ(static_cast<bst_feature_t>(weight[i + beg]) % n_features, 1);
  }
}

TEST(HistUtil, DeviceSketchMultipleColumns) {
  auto ctx = MakeCUDACtx(0);
  auto bin_sizes = {2, 16, 256, 512};
  auto sizes = {100, 1000, 1500};
  int num_columns = 5;
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
    for (auto num_bins : bin_sizes) {
      auto cuts = DeviceSketch(&ctx, dmat.get(), num_bins);
      ValidateCuts(cuts, dmat.get(), num_bins);
    }
  }
}

TEST(HistUtil, DeviceSketchMultipleColumnsWeights) {
  auto ctx = MakeCUDACtx(0);
  auto bin_sizes = {2, 16, 256, 512};
  auto sizes = {100, 1000, 1500};
  int num_columns = 5;
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
    dmat->Info().weights_.HostVector() = GenerateRandomWeights(num_rows);
    for (auto num_bins : bin_sizes) {
      auto cuts = DeviceSketch(&ctx, dmat.get(), num_bins);
      ValidateCuts(cuts, dmat.get(), num_bins);
    }
  }
}

TEST(HistUitl, DeviceSketchWeights) {
  auto ctx = MakeCUDACtx(0);
  auto bin_sizes = {2, 16, 256, 512};
  auto sizes = {100, 1000, 1500};
  int num_columns = 5;
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
    auto weighted_dmat = GetDMatrixFromData(x, num_rows, num_columns);
    auto& h_weights = weighted_dmat->Info().weights_.HostVector();
    h_weights.resize(num_rows);
    std::fill(h_weights.begin(), h_weights.end(), 1.0f);
    for (auto num_bins : bin_sizes) {
      auto cuts = DeviceSketch(&ctx, dmat.get(), num_bins);
      auto wcuts = DeviceSketch(&ctx, weighted_dmat.get(), num_bins);
      ASSERT_EQ(cuts.MinValues(), wcuts.MinValues());
      ASSERT_EQ(cuts.Ptrs(), wcuts.Ptrs());
      ASSERT_EQ(cuts.Values(), wcuts.Values());
      ValidateCuts(cuts, dmat.get(), num_bins);
      ValidateCuts(wcuts, weighted_dmat.get(), num_bins);
    }
  }
}

TEST(HistUtil, DeviceSketchBatches) {
  auto ctx = MakeCUDACtx(0);
  int num_bins = 256;
  int num_rows = 5000;
  auto batch_sizes = {0, 100, 1500, 6000};
  int num_columns = 5;
  for (auto batch_size : batch_sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
    auto cuts = DeviceSketch(&ctx, dmat.get(), num_bins, batch_size);
    ValidateCuts(cuts, dmat.get(), num_bins);
  }

  num_rows = 1000;
  size_t batches = 16;
  auto x = GenerateRandom(num_rows * batches, num_columns);
  auto dmat = GetDMatrixFromData(x, num_rows * batches, num_columns);
  auto cuts_with_batches = DeviceSketch(&ctx, dmat.get(), num_bins, num_rows);
  auto cuts = DeviceSketch(&ctx, dmat.get(), num_bins, 0);

  auto const& cut_values_batched = cuts_with_batches.Values();
  auto const& cut_values = cuts.Values();
  CHECK_EQ(cut_values.size(), cut_values_batched.size());
  for (size_t i = 0; i < cut_values.size(); ++i) {
    ASSERT_NEAR(cut_values_batched[i], cut_values[i], 1e5);
  }
}

TEST(HistUtil, DeviceSketchMultipleColumnsExternal) {
  auto ctx = MakeCUDACtx(0);
  auto bin_sizes = {2, 16, 256, 512};
  auto sizes = {100, 1000, 1500};
  int num_columns =5;
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    dmlc::TemporaryDirectory temp;
    auto dmat = GetExternalMemoryDMatrixFromData(x, num_rows, num_columns, temp);
    for (auto num_bins : bin_sizes) {
      auto cuts = DeviceSketch(&ctx, dmat.get(), num_bins);
      ValidateCuts(cuts, dmat.get(), num_bins);
    }
  }
}

// See https://github.com/dmlc/xgboost/issues/5866.
TEST(HistUtil, DeviceSketchExternalMemoryWithWeights) {
  auto ctx = MakeCUDACtx(0);
  auto bin_sizes = {2, 16, 256, 512};
  auto sizes = {100, 1000, 1500};
  int num_columns = 5;
  dmlc::TemporaryDirectory temp;
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    auto dmat = GetExternalMemoryDMatrixFromData(x, num_rows, num_columns, temp);
    dmat->Info().weights_.HostVector() = GenerateRandomWeights(num_rows);
    for (auto num_bins : bin_sizes) {
      auto cuts = DeviceSketch(&ctx, dmat.get(), num_bins);
      ValidateCuts(cuts, dmat.get(), num_bins);
    }
  }
}

template <typename Adapter>
auto MakeUnweightedCutsForTest(Context const* ctx, Adapter adapter, int32_t num_bins, float missing,
                               size_t batch_size = 0) {
  common::HistogramCuts batched_cuts;
  HostDeviceVector<FeatureType> ft;
  SketchContainer sketch_container(ft, num_bins, adapter.NumColumns(), adapter.NumRows(),
                                   DeviceOrd::CUDA(0));
  MetaInfo info;
  AdapterDeviceSketch(ctx, adapter.Value(), num_bins, info, missing, &sketch_container, batch_size);
  sketch_container.MakeCuts(ctx, &batched_cuts, info.IsColumnSplit());
  return batched_cuts;
}

template <typename Adapter>
void ValidateBatchedCuts(Context const* ctx, Adapter adapter, int num_bins, DMatrix* dmat, size_t batch_size = 0) {
  common::HistogramCuts batched_cuts = MakeUnweightedCutsForTest(
      ctx, adapter, num_bins, std::numeric_limits<float>::quiet_NaN(), batch_size);
  ValidateCuts(batched_cuts, dmat, num_bins);
}

TEST(HistUtil, AdapterDeviceSketch) {
  auto ctx = MakeCUDACtx(0);
  int rows = 5;
  int cols = 1;
  int num_bins = 4;
  float missing =  - 1.0;
  thrust::device_vector< float> data(rows*cols);
  auto json_array_interface = Generate2dArrayInterface(rows, cols, "<f4", &data);
  data = std::vector<float >{ 1.0,2.0,3.0,4.0,5.0 };
  std::string str;
  Json::Dump(json_array_interface, &str);

  data::CupyAdapter adapter(str);

  auto device_cuts = MakeUnweightedCutsForTest(&ctx, adapter, num_bins, missing);
  ctx = ctx.MakeCPU();
  auto host_cuts = GetHostCuts(&ctx, &adapter, num_bins, missing);

  EXPECT_EQ(device_cuts.Values(), host_cuts.Values());
  EXPECT_EQ(device_cuts.Ptrs(), host_cuts.Ptrs());
  EXPECT_EQ(device_cuts.MinValues(), host_cuts.MinValues());
}

TEST(HistUtil, AdapterDeviceSketchMemory) {
  auto ctx = MakeCUDACtx(0);
  int num_columns = 100;
  int num_rows = 1000;
  int num_bins = 256;
  auto x = GenerateRandom(num_rows, num_columns);
  auto x_device = thrust::device_vector<float>(x);
  auto adapter = AdapterFromData(x_device, num_rows, num_columns);

  dh::GlobalMemoryLogger().Clear();
  ConsoleLogger::Configure({{"verbosity", "3"}});
  auto cuts =
      MakeUnweightedCutsForTest(&ctx, adapter, num_bins, std::numeric_limits<float>::quiet_NaN());
  ConsoleLogger::Configure({{"verbosity", "0"}});
  size_t bytes_required = detail::RequiredMemory(
      num_rows, num_columns, num_rows * num_columns, num_bins, false);
  EXPECT_LE(dh::GlobalMemoryLogger().PeakMemory(), bytes_required * 1.05);
  EXPECT_GE(dh::GlobalMemoryLogger().PeakMemory(), bytes_required * 0.95);
}

TEST(HistUtil, AdapterSketchSlidingWindowMemory) {
  auto ctx = MakeCUDACtx(0);
  int num_columns = 100;
  int num_rows = 1000;
  int num_bins = 256;
  auto x = GenerateRandom(num_rows, num_columns);
  auto x_device = thrust::device_vector<float>(x);
  auto adapter = AdapterFromData(x_device, num_rows, num_columns);
  MetaInfo info;

  dh::GlobalMemoryLogger().Clear();
  ConsoleLogger::Configure({{"verbosity", "3"}});
  common::HistogramCuts batched_cuts;
  HostDeviceVector<FeatureType> ft;
  SketchContainer sketch_container(ft, num_bins, num_columns, num_rows, DeviceOrd::CUDA(0));
  AdapterDeviceSketch(&ctx, adapter.Value(), num_bins, info,
                      std::numeric_limits<float>::quiet_NaN(), &sketch_container);
  HistogramCuts cuts;
  sketch_container.MakeCuts(&ctx, &cuts, info.IsColumnSplit());
  size_t bytes_required = detail::RequiredMemory(
      num_rows, num_columns, num_rows * num_columns, num_bins, false);
  EXPECT_LE(dh::GlobalMemoryLogger().PeakMemory(), bytes_required * 1.05);
  EXPECT_GE(dh::GlobalMemoryLogger().PeakMemory(), bytes_required * 0.95);
  ConsoleLogger::Configure({{"verbosity", "0"}});
}

TEST(HistUtil, AdapterSketchSlidingWindowWeightedMemory) {
  auto ctx = MakeCUDACtx(0);
  int num_columns = 100;
  int num_rows = 1000;
  int num_bins = 256;
  auto x = GenerateRandom(num_rows, num_columns);
  auto x_device = thrust::device_vector<float>(x);
  auto adapter = AdapterFromData(x_device, num_rows, num_columns);
  MetaInfo info;
  auto& h_weights = info.weights_.HostVector();
  h_weights.resize(num_rows);
  std::fill(h_weights.begin(), h_weights.end(), 1.0f);

  dh::GlobalMemoryLogger().Clear();
  ConsoleLogger::Configure({{"verbosity", "3"}});
  common::HistogramCuts batched_cuts;
  HostDeviceVector<FeatureType> ft;
  SketchContainer sketch_container(ft, num_bins, num_columns, num_rows, DeviceOrd::CUDA(0));
  AdapterDeviceSketch(&ctx, adapter.Value(), num_bins, info,
                      std::numeric_limits<float>::quiet_NaN(), &sketch_container);

  HistogramCuts cuts;
  sketch_container.MakeCuts(&ctx, &cuts, info.IsColumnSplit());
  ConsoleLogger::Configure({{"verbosity", "0"}});
  size_t bytes_required = detail::RequiredMemory(
      num_rows, num_columns, num_rows * num_columns, num_bins, true);
  EXPECT_LE(dh::GlobalMemoryLogger().PeakMemory(), bytes_required * 1.05);
  EXPECT_GE(dh::GlobalMemoryLogger().PeakMemory(), bytes_required);
}

void TestCategoricalSketchAdapter(size_t n, size_t num_categories,
                                  int32_t num_bins, bool weighted) {
  auto ctx = MakeCUDACtx(0);
  auto h_x = GenerateRandomCategoricalSingleColumn(n, num_categories);
  thrust::device_vector<float> x(h_x);
  auto adapter = AdapterFromData(x, n, 1);
  MetaInfo info;
  info.num_row_ = n;
  info.num_col_ = 1;
  info.feature_types.HostVector().push_back(FeatureType::kCategorical);

  if (weighted) {
    std::vector<float> weights(n, 0);
    SimpleLCG lcg;
    SimpleRealUniformDistribution<float> dist(0, 1);
    for (auto& v : weights) {
      v = dist(&lcg);
    }
    info.weights_.HostVector() = weights;
  }

  ASSERT_EQ(info.feature_types.Size(), 1);
  SketchContainer container(info.feature_types, num_bins, 1, n, DeviceOrd::CUDA(0));
  AdapterDeviceSketch(&ctx, adapter.Value(), num_bins, info,
                      std::numeric_limits<float>::quiet_NaN(), &container);
  HistogramCuts cuts;
  container.MakeCuts(&ctx, &cuts, info.IsColumnSplit());

  thrust::sort(x.begin(), x.end());
  auto n_uniques = thrust::unique(x.begin(), x.end()) - x.begin();
  ASSERT_NE(n_uniques, x.size());
  ASSERT_EQ(cuts.TotalBins(), n_uniques);
  ASSERT_EQ(n_uniques, num_categories);

  auto& values = cuts.cut_values_.HostVector();
  ASSERT_TRUE(std::is_sorted(values.cbegin(), values.cend()));
  auto is_unique = (std::unique(values.begin(), values.end()) - values.begin()) == n_uniques;
  ASSERT_TRUE(is_unique);

  x.resize(n_uniques);
  h_x.resize(n_uniques);
  thrust::copy(x.begin(), x.end(), h_x.begin());
  for (decltype(n_uniques) i = 0; i < n_uniques; ++i) {
    ASSERT_EQ(h_x[i], values[i]);
  }
}

TEST(HistUtil, AdapterDeviceSketchCategorical) {
  auto categorical_sizes = {2, 6, 8, 12};
  int num_bins = 256;
  auto ctx = MakeCUDACtx(0);
  auto sizes = {25, 100, 1000};
  for (auto n : sizes) {
    for (auto num_categories : categorical_sizes) {
      auto x = GenerateRandomCategoricalSingleColumn(n, num_categories);
      auto dmat = GetDMatrixFromData(x, n, 1);
      auto x_device = thrust::device_vector<float>(x);
      auto adapter = AdapterFromData(x_device, n, 1);
      ValidateBatchedCuts(&ctx, adapter, num_bins, dmat.get());
      TestCategoricalSketchAdapter(n, num_categories, num_bins, true);
      TestCategoricalSketchAdapter(n, num_categories, num_bins, false);
    }
  }
}

TEST(HistUtil, AdapterDeviceSketchMultipleColumns) {
  auto bin_sizes = {2, 16, 256, 512};
  auto sizes = {100, 1000, 1500};
  int num_columns = 5;
  auto ctx = MakeCUDACtx(0);
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
    auto x_device = thrust::device_vector<float>(x);
    for (auto num_bins : bin_sizes) {
      auto adapter = AdapterFromData(x_device, num_rows, num_columns);
      ValidateBatchedCuts(&ctx, adapter, num_bins, dmat.get());
    }
  }
}

TEST(HistUtil, AdapterDeviceSketchBatches) {
  int num_bins = 256;
  int num_rows = 5000;
  auto batch_sizes = {0, 100, 1500, 6000};
  int num_columns = 5;
  auto ctx = MakeCUDACtx(0);
  for (auto batch_size : batch_sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
    auto x_device = thrust::device_vector<float>(x);
    auto adapter = AdapterFromData(x_device, num_rows, num_columns);
    ValidateBatchedCuts(&ctx, adapter, num_bins, dmat.get(), batch_size);
  }
}

namespace {
auto MakeData(Context const* ctx, std::size_t n_samples, bst_feature_t n_features) {
  curt::SetDevice(ctx->Ordinal());
  auto n = n_samples * n_features;
  std::vector<float> x;
  x.resize(n);

  std::iota(x.begin(), x.end(), 0);
  std::int32_t c{0};
  float missing = n_samples * n_features;
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (i % 5 == 0) {
      x[i] = missing;
      c++;
    }
  }
  thrust::device_vector<float> d_x;
  d_x = x;

  auto n_invalids = n / 10 * 2 + 1;
  auto is_valid = data::IsValidFunctor{missing};
  return std::tuple{x, d_x, n_invalids, is_valid};
}

void TestGetColumnSize(std::size_t n_samples) {
  auto ctx = MakeCUDACtx(0);
  bst_feature_t n_features = 12;
  [[maybe_unused]] auto [x, d_x, n_invalids, is_valid] = MakeData(&ctx, n_samples, n_features);

  auto adapter = AdapterFromData(d_x, n_samples, n_features);
  auto batch = adapter.Value();

  auto batch_iter = dh::MakeTransformIterator<data::COOTuple>(
      thrust::make_counting_iterator(0llu),
      [=] __device__(std::size_t idx) { return batch.GetElement(idx); });

  dh::caching_device_vector<std::size_t> column_sizes_scan;
  column_sizes_scan.resize(n_features + 1);
  std::vector<std::size_t> h_column_size(column_sizes_scan.size());
  std::vector<std::size_t> h_column_size_1(column_sizes_scan.size());

  auto cuctx = ctx.CUDACtx();
  detail::LaunchGetColumnSizeKernel<decltype(batch_iter), true, true>(
      cuctx, ctx.Device(), IterSpan{batch_iter, batch.Size()}, is_valid,
      dh::ToSpan(column_sizes_scan));
  thrust::copy(column_sizes_scan.begin(), column_sizes_scan.end(), h_column_size.begin());

  detail::LaunchGetColumnSizeKernel<decltype(batch_iter), true, false>(
      cuctx, ctx.Device(), IterSpan{batch_iter, batch.Size()}, is_valid,
      dh::ToSpan(column_sizes_scan));
  thrust::copy(column_sizes_scan.begin(), column_sizes_scan.end(), h_column_size_1.begin());
  ASSERT_EQ(h_column_size, h_column_size_1);

  detail::LaunchGetColumnSizeKernel<decltype(batch_iter), false, true>(
      cuctx, ctx.Device(), IterSpan{batch_iter, batch.Size()}, is_valid,
      dh::ToSpan(column_sizes_scan));
  thrust::copy(column_sizes_scan.begin(), column_sizes_scan.end(), h_column_size_1.begin());
  ASSERT_EQ(h_column_size, h_column_size_1);

  detail::LaunchGetColumnSizeKernel<decltype(batch_iter), false, false>(
      cuctx, ctx.Device(), IterSpan{batch_iter, batch.Size()}, is_valid,
      dh::ToSpan(column_sizes_scan));
  thrust::copy(column_sizes_scan.begin(), column_sizes_scan.end(), h_column_size_1.begin());
  ASSERT_EQ(h_column_size, h_column_size_1);
}
}  // namespace

TEST(HistUtil, GetColumnSize) {
  bst_idx_t n_samples = 4096;
  TestGetColumnSize(n_samples);
}

// Check sketching from adapter or DMatrix results in the same answer
// Consistency here is useful for testing and user experience
TEST(HistUtil, SketchingEquivalent) {
  auto ctx = MakeCUDACtx(0);
  auto bin_sizes = {2, 16, 256, 512};
  auto sizes = {100, 1000, 1500};
  int num_columns = 5;
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
    for (auto num_bins : bin_sizes) {
      auto dmat_cuts = DeviceSketch(&ctx, dmat.get(), num_bins);
      auto x_device = thrust::device_vector<float>(x);
      auto adapter = AdapterFromData(x_device, num_rows, num_columns);
      common::HistogramCuts adapter_cuts = MakeUnweightedCutsForTest(
          &ctx, adapter, num_bins, std::numeric_limits<float>::quiet_NaN());
      EXPECT_EQ(dmat_cuts.Values(), adapter_cuts.Values());
      EXPECT_EQ(dmat_cuts.Ptrs(), adapter_cuts.Ptrs());
      EXPECT_EQ(dmat_cuts.MinValues(), adapter_cuts.MinValues());

      ValidateBatchedCuts(&ctx, adapter, num_bins, dmat.get());
    }
  }
}

TEST(HistUtil, DeviceSketchFromGroupWeights) {
  auto ctx = MakeCUDACtx(0);
  size_t constexpr kRows = 3000, kCols = 200, kBins = 256;
  size_t constexpr kGroups = 10;
  auto m = RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix();

  // sketch with group weight
  auto& h_weights = m->Info().weights_.HostVector();
  h_weights.resize(kGroups);
  std::fill(h_weights.begin(), h_weights.end(), 1.0f);
  std::vector<bst_group_t> groups(kGroups);
  for (size_t i = 0; i < kGroups; ++i) {
    groups[i] = kRows / kGroups;
  }
  m->SetInfo("group", Make1dInterfaceTest(groups.data(), kGroups));
  HistogramCuts weighted_cuts = DeviceSketch(&ctx, m.get(), kBins, 0);

  // sketch with no weight
  h_weights.clear();
  HistogramCuts cuts = DeviceSketch(&ctx, m.get(), kBins, 0);

  ASSERT_EQ(cuts.Values().size(), weighted_cuts.Values().size());
  ASSERT_EQ(cuts.MinValues().size(), weighted_cuts.MinValues().size());
  ASSERT_EQ(cuts.Ptrs().size(), weighted_cuts.Ptrs().size());

  for (size_t i = 0; i < cuts.Values().size(); ++i) {
    EXPECT_EQ(cuts.Values()[i], weighted_cuts.Values()[i]) << "i:"<< i;
  }
  for (size_t i = 0; i < cuts.MinValues().size(); ++i) {
    ASSERT_EQ(cuts.MinValues()[i], weighted_cuts.MinValues()[i]);
  }
  for (size_t i = 0; i < cuts.Ptrs().size(); ++i) {
    ASSERT_EQ(cuts.Ptrs().at(i), weighted_cuts.Ptrs().at(i));
  }
  ValidateCuts(weighted_cuts, m.get(), kBins);
}

void TestAdapterSketchFromWeights(bool with_group) {
  size_t constexpr kRows = 300, kCols = 20, kBins = 256;
  size_t constexpr kGroups = 10;
  HostDeviceVector<float> storage;
  std::string m = RandomDataGenerator{kRows, kCols, 0}
                      .Device(DeviceOrd::CUDA(0))
                      .GenerateArrayInterface(&storage);
  MetaInfo info;
  auto ctx = MakeCUDACtx(0);
  auto& h_weights = info.weights_.HostVector();
  if (with_group) {
    h_weights.resize(kGroups);
  } else {
    h_weights.resize(kRows);
  }
  std::fill(h_weights.begin(), h_weights.end(), 1.0f);

  std::vector<bst_group_t> groups(kGroups);
  if (with_group) {
    for (size_t i = 0; i < kGroups; ++i) {
      groups[i] = kRows / kGroups;
    }
    info.SetInfo(ctx, "group", Make1dInterfaceTest(groups.data(), kGroups));
  }

  info.weights_.SetDevice(DeviceOrd::CUDA(0));
  info.num_row_ = kRows;
  info.num_col_ = kCols;

  data::CupyAdapter adapter(m);
  auto const& batch = adapter.Value();
  HostDeviceVector<FeatureType> ft;
  SketchContainer sketch_container(ft, kBins, kCols, kRows, DeviceOrd::CUDA(0));
  AdapterDeviceSketch(&ctx, adapter.Value(), kBins, info, std::numeric_limits<float>::quiet_NaN(),
                      &sketch_container);

  common::HistogramCuts cuts;
  sketch_container.MakeCuts(&ctx, &cuts, info.IsColumnSplit());

  auto dmat = GetDMatrixFromData(storage.HostVector(), kRows, kCols);
  if (with_group) {
    dmat->Info().SetInfo(ctx, "group", Make1dInterfaceTest(groups.data(), kGroups));
  }

  dmat->Info().SetInfo(ctx, "weight", Make1dInterfaceTest(h_weights.data(), h_weights.size()));
  dmat->Info().num_col_ = kCols;
  dmat->Info().num_row_ = kRows;
  ASSERT_EQ(cuts.Ptrs().size(), kCols + 1);
  ValidateCuts(cuts, dmat.get(), kBins);

  if (with_group) {
    dmat->Info().weights_ = decltype(dmat->Info().weights_)();  // remove weight
    HistogramCuts non_weighted = DeviceSketch(&ctx, dmat.get(), kBins, 0);
    for (size_t i = 0; i < cuts.Values().size(); ++i) {
      ASSERT_EQ(cuts.Values()[i], non_weighted.Values()[i]);
    }
    for (size_t i = 0; i < cuts.MinValues().size(); ++i) {
      ASSERT_EQ(cuts.MinValues()[i], non_weighted.MinValues()[i]);
    }
    for (size_t i = 0; i < cuts.Ptrs().size(); ++i) {
      ASSERT_EQ(cuts.Ptrs().at(i), non_weighted.Ptrs().at(i));
    }
  }

  if (with_group) {
    common::HistogramCuts weighted;
    auto& h_weights = info.weights_.HostVector();
    h_weights.resize(kGroups);
    // Generate different weight.
    for (size_t i = 0; i < h_weights.size(); ++i) {
      // FIXME(jiamingy): Some entries generated GPU test cannot pass the validate cuts if
      // we use more diverse weights, partially caused by
      // https://github.com/dmlc/xgboost/issues/7946
      h_weights[i] = (i % 2 == 0 ? 1 : 2) / static_cast<float>(kGroups);
    }
    SketchContainer sketch_container{ft, kBins, kCols, kRows, DeviceOrd::CUDA(0)};
    AdapterDeviceSketch(&ctx, adapter.Value(), kBins, info, std::numeric_limits<float>::quiet_NaN(),
                        &sketch_container);
    sketch_container.MakeCuts(&ctx, &weighted, info.IsColumnSplit());
    ValidateCuts(weighted, dmat.get(), kBins);
  }
}

TEST(HistUtil, AdapterSketchFromWeights) {
  TestAdapterSketchFromWeights(false);
  TestAdapterSketchFromWeights(true);
}

namespace {
class DeviceSketchWithHessianTest
    : public ::testing::TestWithParam<std::tuple<bool, bst_idx_t, bst_bin_t>> {
  bst_feature_t n_features_ = 5;
  bst_group_t n_groups_{3};

  auto GenerateHessian(Context const* ctx, bst_idx_t n_samples) const {
    HostDeviceVector<float> hessian;
    auto& h_hess = hessian.HostVector();
    h_hess = GenerateRandomWeights(n_samples);
    std::mt19937 rng(0);
    std::shuffle(h_hess.begin(), h_hess.end(), rng);
    hessian.SetDevice(ctx->Device());
    return hessian;
  }

  void CheckReg(Context const* ctx, std::shared_ptr<DMatrix> p_fmat, bst_bin_t n_bins,
                HostDeviceVector<float> const& hessian, std::vector<float> const& w,
                std::size_t n_elements) const {
    auto const& h_hess = hessian.ConstHostVector();
    {
      auto& h_weight = p_fmat->Info().weights_.HostVector();
      h_weight = w;
    }

    HistogramCuts cuts_hess =
        DeviceSketchWithHessian(ctx, p_fmat.get(), n_bins, hessian.ConstDeviceSpan(), n_elements);
    ValidateCuts(cuts_hess, p_fmat.get(), n_bins);

    // merge hessian
    {
      auto& h_weight = p_fmat->Info().weights_.HostVector();
      ASSERT_EQ(h_weight.size(), h_hess.size());
      for (std::size_t i = 0; i < h_weight.size(); ++i) {
        h_weight[i] = w[i] * h_hess[i];
      }
    }

    HistogramCuts cuts_wh = DeviceSketch(ctx, p_fmat.get(), n_bins, n_elements);
    ValidateCuts(cuts_wh, p_fmat.get(), n_bins);
    ASSERT_EQ(cuts_hess.Values().size(), cuts_wh.Values().size());
    for (std::size_t i = 0; i < cuts_hess.Values().size(); ++i) {
      ASSERT_NEAR(cuts_wh.Values()[i], cuts_hess.Values()[i], kRtEps);
    }

    p_fmat->Info().weights_.HostVector() = w;
  }

 protected:
  Context ctx_ = MakeCUDACtx(0);

  void TestLTR(Context const* ctx, bst_idx_t n_samples, bst_bin_t n_bins,
               std::size_t n_elements) const {
    auto x = GenerateRandom(n_samples, n_features_);

    std::vector<bst_group_t> gptr;
    gptr.resize(n_groups_ + 1, 0);
    gptr[1] = n_samples / n_groups_;
    gptr[2] = n_samples / n_groups_ + gptr[1];
    gptr.back() = n_samples;

    auto hessian = this->GenerateHessian(ctx, n_samples);
    auto const& h_hess = hessian.ConstHostVector();
    auto p_fmat = GetDMatrixFromData(x, n_samples, n_features_);
    p_fmat->Info().group_ptr_ = gptr;

    // test with constant group weight
    std::vector<float> w(n_groups_, 1.0f);
    p_fmat->Info().weights_.HostVector() = w;
    HistogramCuts cuts_hess =
        DeviceSketchWithHessian(ctx, p_fmat.get(), n_bins, hessian.ConstDeviceSpan(), n_elements);
    // make validation easier by converting it into sample weight.
    p_fmat->Info().weights_.HostVector() = h_hess;
    p_fmat->Info().group_ptr_.clear();
    ValidateCuts(cuts_hess, p_fmat.get(), n_bins);
    // restore ltr properties
    p_fmat->Info().weights_.HostVector() = w;
    p_fmat->Info().group_ptr_ = gptr;

    // test with random group weight
    w = GenerateRandomWeights(n_groups_);
    p_fmat->Info().weights_.HostVector() = w;
    cuts_hess =
        DeviceSketchWithHessian(ctx, p_fmat.get(), n_bins, hessian.ConstDeviceSpan(), n_elements);
    // make validation easier by converting it into sample weight.
    p_fmat->Info().weights_.HostVector() = h_hess;
    p_fmat->Info().group_ptr_.clear();
    ValidateCuts(cuts_hess, p_fmat.get(), n_bins);

    // merge hessian with sample weight
    p_fmat->Info().weights_.Resize(n_samples);
    p_fmat->Info().group_ptr_.clear();
    for (std::size_t i = 0; i < h_hess.size(); ++i) {
      auto gidx = dh::SegmentId(Span{gptr.data(), gptr.size()}, i);
      p_fmat->Info().weights_.HostVector()[i] = w[gidx] * h_hess[i];
    }
    auto cuts = DeviceSketch(ctx, p_fmat.get(), n_bins, n_elements);
    ValidateCuts(cuts, p_fmat.get(), n_bins);
    ASSERT_EQ(cuts.Values().size(), cuts_hess.Values().size());
    for (std::size_t i = 0; i < cuts.Values().size(); ++i) {
      EXPECT_NEAR(cuts.Values()[i], cuts_hess.Values()[i], 1e-4f);
    }
  }

  void TestRegression(Context const* ctx, bst_idx_t n_samples, bst_bin_t n_bins,
                      std::size_t n_elements) const {
    auto x = GenerateRandom(n_samples, n_features_);
    auto p_fmat = GetDMatrixFromData(x, n_samples, n_features_);
    std::vector<float> w = GenerateRandomWeights(n_samples);

    auto hessian = this->GenerateHessian(ctx, n_samples);

    this->CheckReg(ctx, p_fmat, n_bins, hessian, w, n_elements);
  }
};

auto MakeParamsForTest() {
  std::vector<bst_idx_t> sizes = {1, 2, 256, 512, 1000, 1500};
  std::vector<bst_bin_t> bin_sizes = {2, 16, 256, 512};
  std::vector<std::tuple<bool, bst_idx_t, bst_bin_t>> configs;
  for (auto n_samples : sizes) {
    for (auto n_bins : bin_sizes) {
      configs.emplace_back(true, n_samples, n_bins);
      configs.emplace_back(false, n_samples, n_bins);
    }
  }
  return configs;
}
}  // namespace

TEST_P(DeviceSketchWithHessianTest, DeviceSketchWithHessian) {
  auto param = GetParam();
  auto n_samples = std::get<1>(param);
  auto n_bins = std::get<2>(param);
  if (std::get<0>(param)) {
    this->TestLTR(&ctx_, n_samples, n_bins, 0);
    this->TestLTR(&ctx_, n_samples, n_bins, 512);
  } else {
    this->TestRegression(&ctx_, n_samples, n_bins, 0);
    this->TestRegression(&ctx_, n_samples, n_bins, 512);
  }
}

INSTANTIATE_TEST_SUITE_P(
    HistUtil, DeviceSketchWithHessianTest, ::testing::ValuesIn(MakeParamsForTest()),
    [](::testing::TestParamInfo<DeviceSketchWithHessianTest::ParamType> const& info) {
      auto task = std::get<0>(info.param) ? "ltr" : "reg";
      auto n_samples = std::to_string(std::get<1>(info.param));
      auto n_bins = std::to_string(std::get<2>(info.param));
      return std::string{task} + "_" + n_samples + "_" + n_bins;
    });
}  // namespace xgboost::common
