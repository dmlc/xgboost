#include <dmlc/filesystem.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>


#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>

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

  auto device_cuts = DeviceSketch(0, dmat.get(), num_bins, 0);
  HistogramCuts host_cuts;
  DenseCuts builder(&host_cuts);
  builder.Build(dmat.get(), num_bins);

  EXPECT_EQ(device_cuts.Values(), host_cuts.Values());
  EXPECT_EQ(device_cuts.Ptrs(), host_cuts.Ptrs());
  EXPECT_EQ(device_cuts.MinValues(), host_cuts.MinValues());
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
      auto cuts = DeviceSketch(0, dmat.get(), num_bins, 0);
      ValidateCuts(cuts, dmat.get(), num_bins);
    }
  }
}

TEST(hist_util, DeviceSketchMultipleColumns) {
  int bin_sizes[] = {2, 16, 256, 512};
  int sizes[] = {100, 1000, 1500};
  int num_columns = 2;
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
    for (auto num_bins : bin_sizes) {
      auto cuts = DeviceSketch(0, dmat.get(), num_bins, 0);
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
  int num_columns = 2;
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    dmlc::TemporaryDirectory temp;
    auto dmat =
        GetExternalMemoryDMatrixFromData(x, num_rows, num_columns, 100, temp);
    for (auto num_bins : bin_sizes) {
      auto cuts = DeviceSketch(0, dmat.get(), num_bins, 0);
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

TEST(hist_util, Benchmark) {
  int num_bins = 256;
  std::vector<int> sizes;
  for (auto i = 8ull; i < 26; i += 2) {
    sizes.push_back(1 << i);
  }

  std::cout << "Num rows, ";
  for (auto n : sizes) {
    std::cout << n << ", ";
  }
  std::cout << "\n";
  int num_columns = 5;
  std::cout << "AdapterDeviceSketch, ";
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    auto x_device = thrust::device_vector<float>(x);
    auto adapter = AdapterFromData(x_device, num_rows, num_columns);
    Timer t;
    t.Start();
    auto cuts = AdapterDeviceSketch(&adapter, num_bins,
                                    std::numeric_limits<float>::quiet_NaN());
    t.Stop();
    std::cout << t.ElapsedSeconds() << ", ";
  }
  std::cout << "\n";

  std::cout << "DeviceSketch, ";
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    dmlc::TemporaryDirectory tmpdir;
    auto dmat =
      GetDMatrixFromData(x, num_rows, num_columns);
    Timer t;
    t.Start();
    auto cuts = DeviceSketch(0, dmat.get(), num_bins, 0);
    t.Stop();
    std::cout << t.ElapsedSeconds() << ", ";
  }
  std::cout << "\n";

  std::cout << "WQSketch, ";
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    dmlc::TemporaryDirectory tmpdir;
    auto dmat =
      GetDMatrixFromData(x, num_rows, num_columns);
      HistogramCuts cuts;
      DenseCuts dense(&cuts);
      Timer t;
      t.Start();
      dense.Build(dmat.get(), num_bins);
      t.Stop();
      std::cout << t.ElapsedSeconds() << ", ";
  }
  std::cout << "\n";
}
TEST(hist_util, BenchmarkNumColumns) {
  int num_bins = 256;
  int num_rows = 10;
  std::vector<int> num_columns;
  for (auto i = 4ull; i < 16; i += 2) {
    num_columns.push_back(1 << i);
  }

  std::cout << "Num columns, ";
  for (auto n : num_columns) {
    std::cout << n << ", ";
  }
  std::cout << "\n";
  std::cout << "AdapterDeviceSketch, ";
  for (auto num_column : num_columns) {
    auto x = GenerateRandom(num_rows, num_column);
    auto x_device = thrust::device_vector<float>(x);
    auto adapter = AdapterFromData(x_device, num_rows, num_column);
    Timer t;
    t.Start();
    auto cuts = AdapterDeviceSketch(&adapter, num_bins,
                                    std::numeric_limits<float>::quiet_NaN());
    t.Stop();
    std::cout << t.ElapsedSeconds() << ", ";
  }
  std::cout << "\n";
  std::cout << "DeviceSketch, ";
  for (auto num_column : num_columns) {
    auto x = GenerateRandom(num_rows, num_column);
    dmlc::TemporaryDirectory tmpdir;
    auto dmat =
      GetDMatrixFromData(x, num_rows, num_column);
    Timer t;
    t.Start();
    auto cuts = DeviceSketch(0, dmat.get(), num_bins, 0);
    t.Stop();
    std::cout << t.ElapsedSeconds() << ", ";
  }
  std::cout << "\n";
  std::cout << "SparseCuts, ";
  for (auto num_column : num_columns) {
    auto x = GenerateRandom(num_rows, num_column);
    dmlc::TemporaryDirectory tmpdir;
    auto dmat =
      GetDMatrixFromData(x, num_rows, num_column);
    HistogramCuts cuts;
    SparseCuts sparse(&cuts);
    Timer t;
    t.Start();
    sparse.Build(dmat.get(), num_bins);
    t.Stop();
    std::cout << t.ElapsedSeconds() << ", ";
  }
  std::cout << "\n";
   std::cout << "DenseCuts, ";
  for (auto num_column : num_columns) {
    auto x = GenerateRandom(num_rows, num_column);
    dmlc::TemporaryDirectory tmpdir;
    auto dmat =
      GetDMatrixFromData(x, num_rows, num_column);
    HistogramCuts cuts;
    DenseCuts dense(&cuts);
    Timer t;
    t.Start();
    dense.Build(dmat.get(), num_bins);
    t.Stop();
    std::cout << t.ElapsedSeconds() << ", ";
  }
  std::cout << "\n";
}
}  // namespace common
}  // namespace xgboost
