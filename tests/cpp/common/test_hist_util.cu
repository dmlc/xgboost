#include <dmlc/filesystem.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>


#include <thrust/device_vector.h>

#include "xgboost/c_api.h"

#include "../../../src/common/device_helpers.cuh"
#include "../../../src/common/hist_util.h"

#include "../helpers.h"
#include <xgboost/data.h>
#include "../../../src/data/device_adapter.cuh"
#include "../data/test_array_interface.h"
#include "../../../src/common/math.h"
#include "../../../src/data/simple_dmatrix.h"
#include "test_hist_util.h"
#include "../../../include/xgboost/logging.h"

namespace xgboost {
namespace common {

template <typename AdapterT>
HistogramCuts GetHostCuts(AdapterT *adapter, int num_bins, float missing) {
  HistogramCuts cuts;
  DenseCuts builder(&cuts);
  data::SimpleDMatrix dmat(adapter, missing, 1);
  builder.Build(&dmat, num_bins);
  return cuts;
}
TEST(hist_util, DeviceSketch) {
  int num_rows = 5;
  int num_columns = 1;
  int num_bins = 4;
  std::vector<float> x = {1.0, 2.0, 3.0, 4.0, 5.0};
  auto dmat = GetDMatrixFromData(x, num_rows, num_columns);

  auto device_cuts = DeviceSketch(0, dmat.get(), num_bins);
  HistogramCuts host_cuts;
  DenseCuts builder(&host_cuts);
  builder.Build(dmat.get(), num_bins);

  EXPECT_EQ(device_cuts.Values(), host_cuts.Values());
  EXPECT_EQ(device_cuts.Ptrs(), host_cuts.Ptrs());
  EXPECT_EQ(device_cuts.MinValues(), host_cuts.MinValues());
}

// Duplicate this function from hist_util.cu so we don't have to expose it in
// header
size_t RequiredSampleCutsTest(int max_bins, size_t num_rows) {
  constexpr int kFactor = 8;
  double eps = 1.0 / (kFactor * max_bins);
  size_t dummy_nlevel;
  size_t num_cuts;
  WQuantileSketch<bst_float, bst_float>::LimitSizeLevel(
    num_rows, eps, &dummy_nlevel, &num_cuts);
  return std::min(num_cuts, num_rows);
}

TEST(hist_util, DeviceSketchMemory) {
  int num_columns = 100;
  int num_rows = 1000;
  int num_bins = 256;
  auto x = GenerateRandom(num_rows, num_columns);
  auto dmat = GetDMatrixFromData(x, num_rows, num_columns);

  ConsoleLogger::Configure({{"verbosity", "3"}});
  auto device_cuts = DeviceSketch(0, dmat.get(), num_bins);
  ConsoleLogger::Configure({{"verbosity", "0"}});

  size_t bytes_num_elements = num_rows * num_columns*sizeof(Entry);
  size_t bytes_cuts = RequiredSampleCutsTest(num_bins, num_rows) * num_columns *
                      sizeof(DenseCuts::WQSketch::Entry);
  size_t bytes_constant = 1000;
  EXPECT_LE(dh::GlobalMemoryLogger().PeakMemory(),
            bytes_num_elements + bytes_cuts + bytes_constant);
}

TEST(hist_util, DeviceSketchMemoryWeights) {
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

  size_t bytes_num_elements =
      num_rows * num_columns * (sizeof(Entry) + sizeof(float));
  size_t bytes_cuts = RequiredSampleCutsTest(num_bins, num_rows) * num_columns *
                      sizeof(DenseCuts::WQSketch::Entry);
  EXPECT_LE(dh::GlobalMemoryLogger().PeakMemory(),
            size_t((bytes_num_elements + bytes_cuts) * 1.05));
}

TEST(hist_util, DeviceSketchDeterminism) {
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

 TEST(hist_util, DeviceSketchCategorical) {
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

TEST(hist_util, DeviceSketchMultipleColumns) {
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

TEST(hist_util, DeviceSketchMultipleColumnsWeights) {
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

TEST(hist_util, DeviceSketchBatches) {
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
}

TEST(hist_util, DeviceSketchMultipleColumnsExternal) {
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

TEST(hist_util, AdapterDeviceSketch)
{
  int rows = 5;
  int cols = 1;
  int num_bins = 4;
  float missing =  - 1.0;
  thrust::device_vector< float> data(rows*cols);
  auto json_array_interface = Generate2dArrayInterface(rows, cols, "<f4", &data);
  data = std::vector<float >{ 1.0,2.0,3.0,4.0,5.0 };
  std::stringstream ss;
  Json::Dump(json_array_interface, &ss);
  std::string str = ss.str();
  data::CupyAdapter adapter(str);

  auto device_cuts = AdapterDeviceSketch(&adapter, num_bins, missing);
  auto host_cuts = GetHostCuts(&adapter, num_bins, missing);

  EXPECT_EQ(device_cuts.Values(), host_cuts.Values());
  EXPECT_EQ(device_cuts.Ptrs(), host_cuts.Ptrs());
  EXPECT_EQ(device_cuts.MinValues(), host_cuts.MinValues());
}

TEST(hist_util, AdapterDeviceSketchMemory) {
  int num_columns = 100;
  int num_rows = 1000;
  int num_bins = 256;
  auto x = GenerateRandom(num_rows, num_columns);
  auto x_device = thrust::device_vector<float>(x);
  auto adapter = AdapterFromData(x_device, num_rows, num_columns);

  dh::GlobalMemoryLogger().Clear();
  ConsoleLogger::Configure({{"verbosity", "3"}});
  auto cuts = AdapterDeviceSketch(&adapter, num_bins,
                                  std::numeric_limits<float>::quiet_NaN());
  ConsoleLogger::Configure({{"verbosity", "0"}});

  size_t bytes_num_elements = num_rows * num_columns * sizeof(Entry);
  size_t bytes_num_columns = (num_columns + 1) * sizeof(size_t);
  size_t bytes_cuts = RequiredSampleCutsTest(num_bins, num_rows) * num_columns *
                      sizeof(DenseCuts::WQSketch::Entry);
  size_t bytes_constant = 1000;
  EXPECT_LE(dh::GlobalMemoryLogger().PeakMemory(),
      bytes_num_elements + bytes_cuts + bytes_num_columns + bytes_constant);
}

 TEST(hist_util, AdapterDeviceSketchCategorical) {
  int categorical_sizes[] = {2, 6, 8, 12};
  int num_bins = 256;
  int sizes[] = {25, 100, 1000};
  for (auto n : sizes) {
    for (auto num_categories : categorical_sizes) {
      auto x = GenerateRandomCategoricalSingleColumn(n, num_categories);
      auto dmat = GetDMatrixFromData(x, n, 1);
      auto x_device = thrust::device_vector<float>(x);
      auto adapter = AdapterFromData(x_device, n, 1);
      auto cuts = AdapterDeviceSketch(&adapter, num_bins,
                                      std::numeric_limits<float>::quiet_NaN());
      ValidateCuts(cuts, dmat.get(), num_bins);
    }
  }
}

TEST(hist_util, AdapterDeviceSketchMultipleColumns) {
  int bin_sizes[] = {2, 16, 256, 512};
  int sizes[] = {100, 1000, 1500};
  int num_columns = 5;
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
    auto x_device = thrust::device_vector<float>(x);
    for (auto num_bins : bin_sizes) {
      auto adapter = AdapterFromData(x_device, num_rows, num_columns);
      auto cuts = AdapterDeviceSketch(&adapter, num_bins,
                                      std::numeric_limits<float>::quiet_NaN());
      ValidateCuts(cuts, dmat.get(), num_bins);
    }
  }
}
TEST(hist_util, AdapterDeviceSketchBatches) {
  int num_bins = 256;
  int num_rows = 5000;
  int batch_sizes[] = {0, 100, 1500, 6000};
  int num_columns = 5;
  for (auto batch_size : batch_sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
    auto x_device = thrust::device_vector<float>(x);
    auto adapter = AdapterFromData(x_device, num_rows, num_columns);
    auto cuts = AdapterDeviceSketch(&adapter, num_bins,
                                    std::numeric_limits<float>::quiet_NaN(),
                                    batch_size);
    ValidateCuts(cuts, dmat.get(), num_bins);
  }
}
}  // namespace common
}  // namespace xgboost
