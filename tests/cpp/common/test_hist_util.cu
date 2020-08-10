#include <dmlc/filesystem.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <thrust/device_vector.h>

#include <xgboost/data.h>
#include <xgboost/c_api.h>

#include "test_hist_util.h"
#include "../helpers.h"
#include "../data/test_array_interface.h"
#include "../../../src/common/device_helpers.cuh"
#include "../../../src/common/hist_util.h"
#include "../../../src/common/hist_util.cuh"
#include "../../../src/data/device_adapter.cuh"
#include "../../../src/common/math.h"
#include "../../../src/data/simple_dmatrix.h"
#include "../../../include/xgboost/logging.h"

namespace xgboost {
namespace common {

template <typename AdapterT>
HistogramCuts GetHostCuts(AdapterT *adapter, int num_bins, float missing) {
  data::SimpleDMatrix dmat(adapter, missing, 1);
  HistogramCuts cuts = SketchOnDMatrix(&dmat, num_bins);
  return cuts;
}

TEST(HistUtil, DeviceSketch) {
  int num_columns = 1;
  int num_bins = 4;
  std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 7.0f, -1.0f};
  int num_rows = x.size();
  auto dmat = GetDMatrixFromData(x, num_rows, num_columns);

  auto device_cuts = DeviceSketch(0, dmat.get(), num_bins);
  HistogramCuts host_cuts = SketchOnDMatrix(dmat.get(), num_bins);

  EXPECT_EQ(device_cuts.Values(), host_cuts.Values());
  EXPECT_EQ(device_cuts.Ptrs(), host_cuts.Ptrs());
  EXPECT_EQ(device_cuts.MinValues(), host_cuts.MinValues());
}

TEST(HistUtil, SketchBatchNumElements) {
  size_t constexpr kCols = 10000;
  int device;
  dh::safe_cuda(cudaGetDevice(&device));
  auto avail = static_cast<size_t>(dh::AvailableMemory(device) * 0.8);
  auto per_elem = detail::BytesPerElement(false);
  auto avail_elem = avail / per_elem;
  size_t rows = avail_elem / kCols * 10;
  auto batch = detail::SketchBatchNumElements(0, rows, kCols, rows * kCols, device, 256, false);
  ASSERT_EQ(batch, avail_elem);
}

TEST(HistUtil, DeviceSketchMemory) {
  int num_columns = 100;
  int num_rows = 1000;
  int num_bins = 256;
  auto x = GenerateRandom(num_rows, num_columns);
  auto dmat = GetDMatrixFromData(x, num_rows, num_columns);

  dh::GlobalMemoryLogger().Clear();
  ConsoleLogger::Configure({{"verbosity", "3"}});
  auto device_cuts = DeviceSketch(0, dmat.get(), num_bins);

  size_t bytes_required = detail::RequiredMemory(
      num_rows, num_columns, num_rows * num_columns, num_bins, false);
  EXPECT_LE(dh::GlobalMemoryLogger().PeakMemory(), bytes_required * 1.05);
  EXPECT_GE(dh::GlobalMemoryLogger().PeakMemory(), bytes_required * 0.95);
  ConsoleLogger::Configure({{"verbosity", "0"}});
}

TEST(HistUtil, DeviceSketchWeightsMemory) {
  int num_columns = 100;
  int num_rows = 1000;
  int num_bins = 256;
  auto x = GenerateRandom(num_rows, num_columns);
  auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
  dmat->Info().weights_.HostVector() = GenerateRandomWeights(num_rows);

  dh::GlobalMemoryLogger().Clear();
  ConsoleLogger::Configure({{"verbosity", "3"}});
  auto device_cuts = DeviceSketch(0, dmat.get(), num_bins);
  ConsoleLogger::Configure({{"verbosity", "0"}});

  size_t bytes_required = detail::RequiredMemory(
      num_rows, num_columns, num_rows * num_columns, num_bins, true);
  EXPECT_LE(dh::GlobalMemoryLogger().PeakMemory(), bytes_required * 1.05);
  EXPECT_GE(dh::GlobalMemoryLogger().PeakMemory(), bytes_required);
}

TEST(HistUtil, DeviceSketchDeterminism) {
  int num_rows = 500;
  int num_columns = 5;
  int num_bins = 256;
  auto x = GenerateRandom(num_rows, num_columns);
  auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
  auto reference_sketch = DeviceSketch(0, dmat.get(), num_bins);
  size_t constexpr kRounds{ 100 };
  for (size_t r = 0; r < kRounds; ++r) {
    auto new_sketch = DeviceSketch(0, dmat.get(), num_bins);
    ASSERT_EQ(reference_sketch.Values(), new_sketch.Values());
    ASSERT_EQ(reference_sketch.MinValues(), new_sketch.MinValues());
  }
}

TEST(HistUtil, DeviceSketchCategorical) {
  int categorical_sizes[] = {2, 6, 8, 12};
  int num_bins = 256;
  int sizes[] = {25, 100, 1000};
  for (auto n : sizes) {
    for (auto num_categories : categorical_sizes) {
      auto x = GenerateRandomCategoricalSingleColumn(n, num_categories);
      auto dmat = GetDMatrixFromData(x, n, 1);
      auto cuts = DeviceSketch(0, dmat.get(), num_bins);
      ValidateCuts(cuts, dmat.get(), num_bins);
    }
  }
}

TEST(HistUtil, DeviceSketchMultipleColumns) {
  int bin_sizes[] = {2, 16, 256, 512};
  int sizes[] = {100, 1000, 1500};
  int num_columns = 5;
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
    for (auto num_bins : bin_sizes) {
      auto cuts = DeviceSketch(0, dmat.get(), num_bins);
      ValidateCuts(cuts, dmat.get(), num_bins);
    }
  }
}

TEST(HistUtil, DeviceSketchMultipleColumnsWeights) {
  int bin_sizes[] = {2, 16, 256, 512};
  int sizes[] = {100, 1000, 1500};
  int num_columns = 5;
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
    dmat->Info().weights_.HostVector() = GenerateRandomWeights(num_rows);
    for (auto num_bins : bin_sizes) {
      auto cuts = DeviceSketch(0, dmat.get(), num_bins);
      ValidateCuts(cuts, dmat.get(), num_bins);
    }
  }
}

TEST(HistUitl, DeviceSketchWeights) {
  int bin_sizes[] = {2, 16, 256, 512};
  int sizes[] = {100, 1000, 1500};
  int num_columns = 5;
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
    auto weighted_dmat = GetDMatrixFromData(x, num_rows, num_columns);
    auto& h_weights = weighted_dmat->Info().weights_.HostVector();
    h_weights.resize(num_rows);
    std::fill(h_weights.begin(), h_weights.end(), 1.0f);
    for (auto num_bins : bin_sizes) {
      auto cuts = DeviceSketch(0, dmat.get(), num_bins);
      auto wcuts = DeviceSketch(0, weighted_dmat.get(), num_bins);
      ASSERT_EQ(cuts.MinValues(), wcuts.MinValues());
      ASSERT_EQ(cuts.Ptrs(), wcuts.Ptrs());
      ASSERT_EQ(cuts.Values(), wcuts.Values());
      ValidateCuts(cuts, dmat.get(), num_bins);
      ValidateCuts(wcuts, weighted_dmat.get(), num_bins);
    }
  }
}

TEST(HistUtil, DeviceSketchBatches) {
  int num_bins = 256;
  int num_rows = 5000;
  int batch_sizes[] = {0, 100, 1500, 6000};
  int num_columns = 5;
  for (auto batch_size : batch_sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
    auto cuts = DeviceSketch(0, dmat.get(), num_bins, batch_size);
    ValidateCuts(cuts, dmat.get(), num_bins);
  }

  num_rows = 1000;
  size_t batches = 16;
  auto x = GenerateRandom(num_rows * batches, num_columns);
  auto dmat = GetDMatrixFromData(x, num_rows * batches, num_columns);
  auto cuts_with_batches = DeviceSketch(0, dmat.get(), num_bins, num_rows);
  auto cuts = DeviceSketch(0, dmat.get(), num_bins, 0);

  auto const& cut_values_batched = cuts_with_batches.Values();
  auto const& cut_values = cuts.Values();
  CHECK_EQ(cut_values.size(), cut_values_batched.size());
  for (size_t i = 0; i < cut_values.size(); ++i) {
    ASSERT_NEAR(cut_values_batched[i], cut_values[i], 1e5);
  }
}

TEST(HistUtil, DeviceSketchMultipleColumnsExternal) {
  int bin_sizes[] = {2, 16, 256, 512};
  int sizes[] = {100, 1000, 1500};
  int num_columns =5;
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    dmlc::TemporaryDirectory temp;
    auto dmat =
        GetExternalMemoryDMatrixFromData(x, num_rows, num_columns, 100, temp);
    for (auto num_bins : bin_sizes) {
      auto cuts = DeviceSketch(0, dmat.get(), num_bins);
      ValidateCuts(cuts, dmat.get(), num_bins);
    }
  }
}

// See https://github.com/dmlc/xgboost/issues/5866.
TEST(HistUtil, DeviceSketchExternalMemoryWithWeights) {
  int bin_sizes[] = {2, 16, 256, 512};
  int sizes[] = {100, 1000, 1500};
  int num_columns = 5;
  dmlc::TemporaryDirectory temp;
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    auto dmat = GetExternalMemoryDMatrixFromData(x, num_rows, num_columns, 100, temp);
    dmat->Info().weights_.HostVector() = GenerateRandomWeights(num_rows);
    for (auto num_bins : bin_sizes) {
      auto cuts = DeviceSketch(0, dmat.get(), num_bins);
      ValidateCuts(cuts, dmat.get(), num_bins);
    }
  }
}

template <typename Adapter>
auto MakeUnweightedCutsForTest(Adapter adapter, int32_t num_bins, float missing, size_t batch_size = 0) {
  common::HistogramCuts batched_cuts;
  SketchContainer sketch_container(num_bins, adapter.NumColumns(), adapter.NumRows(), 0);
  MetaInfo info;
  AdapterDeviceSketch(adapter.Value(), num_bins, info, std::numeric_limits<float>::quiet_NaN(),
                      &sketch_container);
  sketch_container.MakeCuts(&batched_cuts);
  return batched_cuts;
}

template <typename Adapter>
void ValidateBatchedCuts(Adapter adapter, int num_bins, int num_columns, int num_rows,
                         DMatrix* dmat, size_t batch_size = 0) {
  common::HistogramCuts batched_cuts = MakeUnweightedCutsForTest(
      adapter, num_bins, std::numeric_limits<float>::quiet_NaN());
  ValidateCuts(batched_cuts, dmat, num_bins);
}

TEST(HistUtil, AdapterDeviceSketch) {
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

  auto device_cuts = MakeUnweightedCutsForTest(adapter, num_bins, missing);
  auto host_cuts = GetHostCuts(&adapter, num_bins, missing);

  EXPECT_EQ(device_cuts.Values(), host_cuts.Values());
  EXPECT_EQ(device_cuts.Ptrs(), host_cuts.Ptrs());
  EXPECT_EQ(device_cuts.MinValues(), host_cuts.MinValues());
}

TEST(HistUtil, AdapterDeviceSketchMemory) {
  int num_columns = 100;
  int num_rows = 1000;
  int num_bins = 256;
  auto x = GenerateRandom(num_rows, num_columns);
  auto x_device = thrust::device_vector<float>(x);
  auto adapter = AdapterFromData(x_device, num_rows, num_columns);

  dh::GlobalMemoryLogger().Clear();
  ConsoleLogger::Configure({{"verbosity", "3"}});
  auto cuts = MakeUnweightedCutsForTest(adapter, num_bins, std::numeric_limits<float>::quiet_NaN());
  ConsoleLogger::Configure({{"verbosity", "0"}});
  size_t bytes_constant = 1000;
  size_t bytes_required = detail::RequiredMemory(
      num_rows, num_columns, num_rows * num_columns, num_bins, false);
  EXPECT_LE(dh::GlobalMemoryLogger().PeakMemory(), bytes_required + bytes_constant);
  EXPECT_GE(dh::GlobalMemoryLogger().PeakMemory(), bytes_required * 0.95);
}

TEST(HistUtil, AdapterSketchSlidingWindowMemory) {
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
  SketchContainer sketch_container(num_bins, num_columns, num_rows, 0);
  AdapterDeviceSketch(adapter.Value(), num_bins, info, std::numeric_limits<float>::quiet_NaN(),
                      &sketch_container);
  HistogramCuts cuts;
  sketch_container.MakeCuts(&cuts);
  size_t bytes_required = detail::RequiredMemory(
      num_rows, num_columns, num_rows * num_columns, num_bins, false);
  EXPECT_LE(dh::GlobalMemoryLogger().PeakMemory(), bytes_required * 1.05);
  EXPECT_GE(dh::GlobalMemoryLogger().PeakMemory(), bytes_required * 0.95);
  ConsoleLogger::Configure({{"verbosity", "0"}});
}

TEST(HistUtil, AdapterSketchSlidingWindowWeightedMemory) {
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
  SketchContainer sketch_container(num_bins, num_columns, num_rows, 0);
  AdapterDeviceSketch(adapter.Value(), num_bins, info,
                      std::numeric_limits<float>::quiet_NaN(),
                      &sketch_container);
  HistogramCuts cuts;
  sketch_container.MakeCuts(&cuts);
  ConsoleLogger::Configure({{"verbosity", "0"}});
  size_t bytes_required = detail::RequiredMemory(
      num_rows, num_columns, num_rows * num_columns, num_bins, true);
  EXPECT_LE(dh::GlobalMemoryLogger().PeakMemory(), bytes_required * 1.05);
  EXPECT_GE(dh::GlobalMemoryLogger().PeakMemory(), bytes_required);
}

TEST(HistUtil, AdapterDeviceSketchCategorical) {
  int categorical_sizes[] = {2, 6, 8, 12};
  int num_bins = 256;
  int sizes[] = {25, 100, 1000};
  for (auto n : sizes) {
    for (auto num_categories : categorical_sizes) {
      auto x = GenerateRandomCategoricalSingleColumn(n, num_categories);
      auto dmat = GetDMatrixFromData(x, n, 1);
      auto x_device = thrust::device_vector<float>(x);
      auto adapter = AdapterFromData(x_device, n, 1);
      ValidateBatchedCuts(adapter, num_bins, adapter.NumColumns(),
                          adapter.NumRows(), dmat.get());
    }
  }
}

TEST(HistUtil, AdapterDeviceSketchMultipleColumns) {
  int bin_sizes[] = {2, 16, 256, 512};
  int sizes[] = {100, 1000, 1500};
  int num_columns = 5;
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
    auto x_device = thrust::device_vector<float>(x);
    for (auto num_bins : bin_sizes) {
      auto adapter = AdapterFromData(x_device, num_rows, num_columns);
      ValidateBatchedCuts(adapter, num_bins, num_columns, num_rows, dmat.get());
    }
  }
}

TEST(HistUtil, AdapterDeviceSketchBatches) {
  int num_bins = 256;
  int num_rows = 5000;
  int batch_sizes[] = {0, 100, 1500, 6000};
  int num_columns = 5;
  for (auto batch_size : batch_sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
    auto x_device = thrust::device_vector<float>(x);
    auto adapter = AdapterFromData(x_device, num_rows, num_columns);
    ValidateBatchedCuts(adapter, num_bins, num_columns, num_rows, dmat.get(), batch_size);
  }
}

// Check sketching from adapter or DMatrix results in the same answer
// Consistency here is useful for testing and user experience
TEST(HistUtil, SketchingEquivalent) {
  int bin_sizes[] = {2, 16, 256, 512};
  int sizes[] = {100, 1000, 1500};
  int num_columns = 5;
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
    for (auto num_bins : bin_sizes) {
      auto dmat_cuts = DeviceSketch(0, dmat.get(), num_bins);
      auto x_device = thrust::device_vector<float>(x);
      auto adapter = AdapterFromData(x_device, num_rows, num_columns);
      common::HistogramCuts adapter_cuts = MakeUnweightedCutsForTest(
          adapter, num_bins, std::numeric_limits<float>::quiet_NaN());
      EXPECT_EQ(dmat_cuts.Values(), adapter_cuts.Values());
      EXPECT_EQ(dmat_cuts.Ptrs(), adapter_cuts.Ptrs());
      EXPECT_EQ(dmat_cuts.MinValues(), adapter_cuts.MinValues());

      ValidateBatchedCuts(adapter, num_bins, num_columns, num_rows, dmat.get());
    }
  }
}

TEST(HistUtil, DeviceSketchFromGroupWeights) {
  size_t constexpr kRows = 3000, kCols = 200, kBins = 256;
  size_t constexpr kGroups = 10;
  auto m = RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix();
  auto& h_weights = m->Info().weights_.HostVector();
  h_weights.resize(kRows);
  std::fill(h_weights.begin(), h_weights.end(), 1.0f);
  std::vector<bst_group_t> groups(kGroups);
  for (size_t i = 0; i < kGroups; ++i) {
    groups[i] = kRows / kGroups;
  }
  m->Info().SetInfo("group", groups.data(), DataType::kUInt32, kGroups);
  HistogramCuts weighted_cuts = DeviceSketch(0, m.get(), kBins, 0);

  h_weights.clear();
  HistogramCuts cuts = DeviceSketch(0, m.get(), kBins, 0);

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
  std::string m =
      RandomDataGenerator{kRows, kCols, 0}.Device(0).GenerateArrayInterface(
          &storage);
  MetaInfo info;
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
    info.SetInfo("group", groups.data(), DataType::kUInt32, kGroups);
  }

  info.weights_.SetDevice(0);
  info.num_row_ = kRows;
  info.num_col_ = kCols;

  data::CupyAdapter adapter(m);
  auto const& batch = adapter.Value();
  SketchContainer sketch_container(kBins, kCols, kRows, 0);
  AdapterDeviceSketch(adapter.Value(), kBins, info, std::numeric_limits<float>::quiet_NaN(),
                      &sketch_container);
  common::HistogramCuts cuts;
  sketch_container.MakeCuts(&cuts);

  auto dmat = GetDMatrixFromData(storage.HostVector(), kRows, kCols);
  if (with_group) {
    dmat->Info().SetInfo("group", groups.data(), DataType::kUInt32, kGroups);
  }

  dmat->Info().SetInfo("weight", h_weights.data(), DataType::kFloat32, h_weights.size());
  dmat->Info().num_col_ = kCols;
  dmat->Info().num_row_ = kRows;
  ASSERT_EQ(cuts.Ptrs().size(), kCols + 1);
  ValidateCuts(cuts, dmat.get(), kBins);

  if (with_group) {
    HistogramCuts non_weighted = DeviceSketch(0, dmat.get(), kBins, 0);
    for (size_t i = 0; i < cuts.Values().size(); ++i) {
      EXPECT_EQ(cuts.Values()[i], non_weighted.Values()[i]);
    }
    for (size_t i = 0; i < cuts.MinValues().size(); ++i) {
      ASSERT_EQ(cuts.MinValues()[i], non_weighted.MinValues()[i]);
    }
    for (size_t i = 0; i < cuts.Ptrs().size(); ++i) {
      ASSERT_EQ(cuts.Ptrs().at(i), non_weighted.Ptrs().at(i));
    }
  }
}

TEST(HistUtil, AdapterSketchFromWeights) {
  TestAdapterSketchFromWeights(false);
  TestAdapterSketchFromWeights(true);
}
}  // namespace common
}  // namespace xgboost
