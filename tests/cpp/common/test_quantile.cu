#include <gtest/gtest.h>
#include "../helpers.h"
#include "../../../src/common/hist_util.cuh"
#include "../../../src/common/quantile.cuh"

namespace xgboost {
namespace common {
TEST(GPUQuantile, Basic) {
  constexpr size_t kRows = 1000, kCols = 100, kBins = 256;
  SketchContainer sketch(kBins, kCols, kRows, 0);
  dh::caching_device_vector<SketchEntry> entries;
  dh::device_vector<bst_row_t> cuts_ptr(kCols+1);
  thrust::fill(cuts_ptr.begin(), cuts_ptr.end(), 0);
  // Push empty
  sketch.Push(dh::ToSpan(cuts_ptr), &entries);
  ASSERT_EQ(sketch.Data().size(), 0);
}

void TestSketchUnique(float sparsity) {
  constexpr size_t kRows = 1000, kCols = 100, kBins = 256;
  SketchContainer sketch(kBins, kCols, kRows, 0);

  HostDeviceVector<float> storage;
  std::string interface_str = RandomDataGenerator{kRows, kCols, sparsity}
                                  .Device(0)
                                  .GenerateArrayInterface(&storage);
  data::CupyAdapter adapter(interface_str);
  AdapterDeviceSketch(adapter.Value(), kBins,
                      std::numeric_limits<float>::quiet_NaN(), &sketch);
  auto n_cuts = detail::RequiredSampleCutsPerColumn(kBins, kRows);
  if (sparsity == 0) {
    ASSERT_EQ(sketch.Data().size(), n_cuts * kCols);
  } else {
    ASSERT_LT(sketch.Data().size(), n_cuts * kCols * (sparsity + 0.05));
    ASSERT_GT(sketch.Data().size(), n_cuts * kCols * (sparsity - 0.05));
  }

  sketch.Unique();
  ASSERT_TRUE(thrust::is_sorted(thrust::device, sketch.Data().data(),
                                sketch.Data().data() + sketch.Data().size(),
                                detail::SketchUnique{}));
}

TEST(GPUQuantile, Unique) {
  TestSketchUnique(0);
  TestSketchUnique(0.5);
}

void TestQuantileElemRank(int32_t device, Span<SketchEntry const> in,
                          Span<bst_row_t const> d_columns_ptr) {
  dh::LaunchN(device, in.size(), [=] __device__(size_t idx) {
    auto column_id = dh::SegmentId(d_columns_ptr, idx);
    auto in_column = in.subspan(d_columns_ptr[column_id],
                                d_columns_ptr[column_id + 1] -
                                    d_columns_ptr[column_id]);
    idx -= d_columns_ptr[column_id];
    float prev_rmin = idx == 0 ? 0.0f : in_column[idx-1].rmin;
    SPAN_CHECK(in_column[idx].rmin >= prev_rmin);
    float prev_rmax = idx == 0 ? 0.0f : in_column[idx-1].rmax;
    SPAN_CHECK(in_column[idx].rmax >= prev_rmax);
    float rmin_next = in_column[idx].RMinNext();
    SPAN_CHECK(in_column[idx].rmax >= rmin_next);
  });
  // Force sync to terminate current test instead of a later one.
  dh::DebugSyncDevice(__FILE__, __LINE__);
}

TEST(GPUQuantile, Prune) {
  constexpr size_t kRows = 1000, kCols = 100, kBins = 256;
  SketchContainer sketch(kBins, kCols, kRows, 0);

  HostDeviceVector<float> storage;
  std::string interface_str =
      RandomDataGenerator{kRows, kCols, 0}.Device(0).GenerateArrayInterface(
          &storage);
  data::CupyAdapter adapter(interface_str);
  AdapterDeviceSketch(adapter.Value(), kBins, std::numeric_limits<float>::quiet_NaN(),
                      &sketch);
  auto n_cuts = detail::RequiredSampleCutsPerColumn(kBins, kRows);
  ASSERT_EQ(sketch.Data().size(), n_cuts * kCols);

  sketch.Prune(kBins);
  ASSERT_EQ(sketch.Data().size(), kBins * kCols);
  // This is not necessarily true for all inputs without calling unique after prune.
  ASSERT_TRUE(thrust::is_sorted(thrust::device,
                                sketch.Data().data(),
                                sketch.Data().data() + sketch.Data().size(),
                                detail::SketchUnique{}));
  TestQuantileElemRank(0, sketch.Data(), sketch.ColumnsPtr());
}

TEST(GPUQuantile, Merge) {
  constexpr size_t kRows = 1000, kCols = 100, kBins = 256;
  SketchContainer sketch_0(kBins, kCols, kRows, 0);
  HostDeviceVector<float> storage_0;
  std::string interface_str_0 =
      RandomDataGenerator{kRows, kCols, 0}.Device(0).GenerateArrayInterface(
          &storage_0);
  data::CupyAdapter adapter_0(interface_str_0);
  AdapterDeviceSketch(adapter_0.Value(), kBins,
                      std::numeric_limits<float>::quiet_NaN(), &sketch_0);

  SketchContainer sketch_1(kBins, kCols, kRows, 0);
  HostDeviceVector<float> storage_1;
  std::string interface_str_1 =
      RandomDataGenerator{kRows, kCols, 0}.Device(0).GenerateArrayInterface(
          &storage_0);
  data::CupyAdapter adapter_1(interface_str_0);
  AdapterDeviceSketch(adapter_1.Value(), kBins,
                      std::numeric_limits<float>::quiet_NaN(), &sketch_1);

  size_t size_before_merge = sketch_0.Data().size();
  sketch_0.Merge(sketch_1.ColumnsPtr(), sketch_1.Data());
  TestQuantileElemRank(0, sketch_0.Data(), sketch_0.ColumnsPtr());
  sketch_0.FixError();
  TestQuantileElemRank(0, sketch_0.Data(), sketch_0.ColumnsPtr());

  auto columns_ptr = sketch_0.ColumnsPtr();
  std::vector<bst_row_t> h_columns_ptr(columns_ptr.size());
  dh::CopyDeviceSpanToVector(&h_columns_ptr, columns_ptr);
  ASSERT_EQ(h_columns_ptr.back(), sketch_1.Data().size() + size_before_merge);

  sketch_0.Unique();
  ASSERT_TRUE(thrust::is_sorted(thrust::device,
                                sketch_0.Data().data(),
                                sketch_0.Data().data() + sketch_0.Data().size(),
                                detail::SketchUnique{}));
}

TEST(GPUQuantile, AllReduce) {
  // This test is supposed to run by a python test that setups the environment.
  std::string msg {"Skipping AllReduce test"};
#if defined(__linux__) && defined(XGBOOST_USE_NCCL)
  constexpr size_t kRows = 1000, kCols = 100, kBins = 256;
  auto n_gpus = AllVisibleGPUs();
  auto port = std::getenv("DMLC_TRACKER_PORT");
  std::string port_str;
  if (port) {
    port_str = port;
  } else {
    LOG(WARNING) << msg << " as `DMLC_TRACKER_PORT` is not set up.";
    return;
  }

  std::vector<std::string> envs{
      "DMLC_TRACKER_PORT=" + port_str,
      "DMLC_TRACKER_URI=127.0.0.1",
      "DMLC_NUM_WORKER=" + std::to_string(n_gpus)};
  char* c_envs[] {&(envs[0][0]), &(envs[1][0]), &(envs[2][0])};
  rabit::Init(3, c_envs);

  // Set up single node version;
  SketchContainer sketch_on_single_node(kBins, kCols, kRows, 0);
  auto world = rabit::GetWorldSize();
  if (world != 1) {
    ASSERT_EQ(world, n_gpus);
  }

  for (auto rank = 0; rank < world; ++rank) {
    HostDeviceVector<float> storage;
    std::string interface_str =
        RandomDataGenerator{kRows, kCols, 0}.Device(0).Seed(rank).GenerateArrayInterface(
            &storage);
    data::CupyAdapter adapter(interface_str);
    AdapterDeviceSketch(adapter.Value(), kBins,
                        std::numeric_limits<float>::quiet_NaN(),
                        &sketch_on_single_node);
  }
  sketch_on_single_node.Unique();

  // Set up distributed version.  We rely on using rank as seed to generate the exact same
  // copy of data.
  auto rank = rabit::GetRank();
  SketchContainer sketch_distributed(kBins, kCols, kRows, 0);
  HostDeviceVector<float> storage;
  std::string interface_str =
      RandomDataGenerator{kRows, kCols, 0}.Device(0).Seed(rank).GenerateArrayInterface(
          &storage);
  data::CupyAdapter adapter(interface_str);
  AdapterDeviceSketch(adapter.Value(), kBins,
                      std::numeric_limits<float>::quiet_NaN(),
                      &sketch_distributed);
  sketch_distributed.AllReduce();
  sketch_distributed.Unique();

  ASSERT_EQ(sketch_distributed.ColumnsPtr().size(), sketch_on_single_node.ColumnsPtr().size());
  ASSERT_EQ(sketch_distributed.Data().size(), sketch_on_single_node.Data().size());

  TestQuantileElemRank(0, sketch_distributed.Data(), sketch_distributed.ColumnsPtr());

  std::vector<SketchEntry> single_node_data(sketch_on_single_node.Data().size());
  dh::CopyDeviceSpanToVector(&single_node_data, sketch_on_single_node.Data());

  std::vector<SketchEntry> distributed_data(sketch_distributed.Data().size());
  dh::CopyDeviceSpanToVector(&distributed_data, sketch_distributed.Data());

  for (size_t i = 0; i < single_node_data.size(); ++i) {
    // It's possible that the result is not exactly equal when running on different
    // platforms like different GPUs.
    ASSERT_EQ(single_node_data[i].value, distributed_data[i].value);
    ASSERT_EQ(single_node_data[i].rmax, distributed_data[i].rmax);
    ASSERT_EQ(single_node_data[i].rmin, distributed_data[i].rmin);
    ASSERT_EQ(single_node_data[i].wmin, distributed_data[i].wmin);
  }

  rabit::Finalize();
#else
  LOG(WARNING) << msg;
  return;
#endif  // !defined(__linux__)
}

}  // namespace common
}  // namespace xgboost
