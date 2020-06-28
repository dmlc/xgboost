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

TEST(GPUQuantile, Unique) {
  constexpr size_t kRows = 1000, kCols = 100, kBins = 256;
  {
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

    sketch.Unique();
    ASSERT_TRUE(thrust::is_sorted(thrust::device,
                                  sketch.Data().data(),
                                  sketch.Data().data() + sketch.Data().size(),
                                  detail::SketchUnique{}));
  }
  {
    SketchContainer sketch(kBins, kCols, kRows, 0);

    HostDeviceVector<float> storage;
    std::string interface_str =
        RandomDataGenerator{kRows, kCols, 0.5}.Device(0).GenerateArrayInterface(
            &storage);
    data::CupyAdapter adapter(interface_str);
    AdapterDeviceSketch(adapter.Value(), kBins, std::numeric_limits<float>::quiet_NaN(),
                        &sketch);
    auto n_cuts = detail::RequiredSampleCutsPerColumn(kBins, kRows);
    ASSERT_LT(sketch.Data().size(), n_cuts * kCols * 0.6);
    ASSERT_GT(sketch.Data().size(), n_cuts * kCols * 0.4);

    sketch.Unique();
    ASSERT_TRUE(thrust::is_sorted(thrust::device,
                                  sketch.Data().data(),
                                  sketch.Data().data() + sketch.Data().size(),
                                  detail::SketchUnique{}));
  }
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
}  // namespace common
}  // namespace xgboost
