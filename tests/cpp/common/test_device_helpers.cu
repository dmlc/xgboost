/*!
 * Copyright 2017-2021 XGBoost contributors
 */
#include <cstddef>
#include <cstdint>
#include <thrust/device_vector.h>
#include <vector>
#include <xgboost/base.h>
#include "../../../src/common/device_helpers.cuh"
#include "../../../src/common/quantile.h"
#include "../helpers.h"
#include "gtest/gtest.h"

TEST(SumReduce, Test) {
  thrust::device_vector<float> data(100, 1.0f);
  auto sum = dh::SumReduction(data.data().get(), data.size());
  ASSERT_NEAR(sum, 100.0f, 1e-5);
}

void TestAtomicSizeT() {
  size_t constexpr kThreads = 235;
  dh::device_vector<size_t> out(1, 0);
  auto d_out = dh::ToSpan(out);
  dh::LaunchN(kThreads, [=] __device__(size_t idx) {
    atomicAdd(&d_out[0], static_cast<size_t>(1));
  });
  ASSERT_EQ(out[0], kThreads);
}

TEST(AtomicAdd, SizeT) {
  TestAtomicSizeT();
}

void TestSegmentID() {
  std::vector<size_t> segments{0, 1, 3};
  thrust::device_vector<size_t> d_segments(segments);
  auto s_segments = dh::ToSpan(d_segments);
  dh::LaunchN(1, [=]__device__(size_t idx) {
    auto id = dh::SegmentId(s_segments, 0);
    SPAN_CHECK(id == 0);
    id = dh::SegmentId(s_segments, 1);
    SPAN_CHECK(id == 1);
    id = dh::SegmentId(s_segments, 2);
    SPAN_CHECK(id == 1);
  });
}

TEST(SegmentID, Basic) {
  TestSegmentID();
}

TEST(SegmentedUnique, Basic) {
  std::vector<float> values{0.1f, 0.2f, 0.3f, 0.62448811531066895f, 0.62448811531066895f, 0.4f};
  std::vector<size_t> segments{0, 3, 6};

  thrust::device_vector<float> d_values(values);
  thrust::device_vector<xgboost::bst_feature_t> d_segments{segments};

  thrust::device_vector<xgboost::bst_feature_t> d_segs_out(d_segments.size());
  thrust::device_vector<float> d_vals_out(d_values.size());

  size_t n_uniques = dh::SegmentedUnique(
      d_segments.data().get(), d_segments.data().get() + d_segments.size(),
      d_values.data().get(), d_values.data().get() + d_values.size(),
      d_segs_out.data().get(), d_vals_out.data().get(),
      thrust::equal_to<float>{});
  CHECK_EQ(n_uniques, 5);

  std::vector<float> values_sol{0.1f, 0.2f, 0.3f, 0.62448811531066895f, 0.4f};
  for (size_t i = 0 ; i < values_sol.size(); i ++) {
    ASSERT_EQ(d_vals_out[i], values_sol[i]);
  }

  std::vector<xgboost::bst_feature_t> segments_sol{0, 3, 5};
  for (size_t i = 0; i < d_segments.size(); ++i) {
    ASSERT_EQ(segments_sol[i], d_segs_out[i]);
  }

  d_segments[1] = 4;
  d_segments[2] = 6;
  n_uniques = dh::SegmentedUnique(
      d_segments.data().get(), d_segments.data().get() + d_segments.size(),
      d_values.data().get(), d_values.data().get() + d_values.size(),
      d_segs_out.data().get(), d_vals_out.data().get(),
      thrust::equal_to<float>{});
  ASSERT_EQ(n_uniques, values.size());
  for (size_t i = 0 ; i < values.size(); i ++) {
    ASSERT_EQ(d_vals_out[i], values[i]);
  }
}

namespace {
using SketchEntry = xgboost::common::WQSummary<float, float>::Entry;
struct SketchUnique {
  bool __device__ operator()(SketchEntry const& a, SketchEntry const& b) const {
    return a.value - b.value == 0;
  }
};
struct IsSorted {
  bool __device__ operator()(SketchEntry const& a, SketchEntry const& b) const {
    return a.value < b.value;
  }
};
}  // namespace

namespace xgboost {
void TestSegmentedUniqueRegression(std::vector<SketchEntry> values, size_t n_duplicated) {
  std::vector<bst_feature_t> segments{0, static_cast<bst_feature_t>(values.size())};

  thrust::device_vector<SketchEntry> d_values(values);
  thrust::device_vector<bst_feature_t> d_segments(segments);
  thrust::device_vector<bst_feature_t> d_segments_out(segments.size());

  size_t n_uniques = dh::SegmentedUnique(
      d_segments.data().get(), d_segments.data().get() + d_segments.size(), d_values.data().get(),
      d_values.data().get() + d_values.size(), d_segments_out.data().get(), d_values.data().get(),
      SketchUnique{});
  ASSERT_EQ(n_uniques, values.size() - n_duplicated);
  ASSERT_TRUE(thrust::is_sorted(thrust::device, d_values.begin(),
                                d_values.begin() + n_uniques, IsSorted{}));
  ASSERT_EQ(segments.at(0), d_segments_out[0]);
  ASSERT_EQ(segments.at(1), d_segments_out[1] + n_duplicated);
}

TEST(DeviceHelpers, Reduce) {
  size_t kSize = std::numeric_limits<uint32_t>::max();
  auto it = thrust::make_counting_iterator(0ul);
  dh::XGBCachingDeviceAllocator<char> alloc;
  auto batched = dh::Reduce(thrust::cuda::par(alloc), it, it + kSize, 0ul, thrust::maximum<size_t>{});
  CHECK_EQ(batched, kSize - 1);
}


TEST(SegmentedUnique, Regression) {
  {
    std::vector<SketchEntry> values{{3149, 3150, 1, 0.62392902374267578},
                                    {3151, 3152, 1, 0.62418866157531738},
                                    {3152, 3153, 1, 0.62419462203979492},
                                    {3153, 3154, 1, 0.62431186437606812},
                                    {3154, 3155, 1, 0.6244881153106689453125},
                                    {3155, 3156, 1, 0.6244881153106689453125},
                                    {3155, 3156, 1, 0.6244881153106689453125},
                                    {3155, 3156, 1, 0.6244881153106689453125},
                                    {3157, 3158, 1, 0.62552797794342041},
                                    {3158, 3159, 1, 0.6256556510925293},
                                    {3159, 3160, 1, 0.62571090459823608},
                                    {3160, 3161, 1, 0.62577134370803833}};
    TestSegmentedUniqueRegression(values, 3);
  }
  {
    std::vector<SketchEntry> values{{3149, 3150, 1, 0.62392902374267578},
                                    {3151, 3152, 1, 0.62418866157531738},
                                    {3152, 3153, 1, 0.62419462203979492},
                                    {3153, 3154, 1, 0.62431186437606812},
                                    {3154, 3155, 1, 0.6244881153106689453125},
                                    {3157, 3158, 1, 0.62552797794342041},
                                    {3158, 3159, 1, 0.6256556510925293},
                                    {3159, 3160, 1, 0.62571090459823608},
                                    {3160, 3161, 1, 0.62577134370803833}};
    TestSegmentedUniqueRegression(values, 0);
  }
  {
    std::vector<SketchEntry> values;
    TestSegmentedUniqueRegression(values, 0);
  }
}

TEST(Allocator, OOM) {
  auto size = dh::AvailableMemory(0) * 4;
  ASSERT_THROW({dh::caching_device_vector<char> vec(size);}, dmlc::Error);
  ASSERT_THROW({dh::device_vector<char> vec(size);}, dmlc::Error);
  // Clear last error so we don't fail subsequent tests
  cudaGetLastError();
}

TEST(DeviceHelpers, ArgSort) {
  dh::device_vector<float> values(20);
  dh::Iota(dh::ToSpan(values));  // accending
  dh::device_vector<size_t> sorted_idx(20);
  dh::ArgSort<false>(dh::ToSpan(values), dh::ToSpan(sorted_idx));  // sort to descending
  ASSERT_TRUE(thrust::is_sorted(thrust::device, sorted_idx.begin(),
                                sorted_idx.end(), thrust::greater<size_t>{}));

  dh::Iota(dh::ToSpan(values));
  dh::device_vector<size_t> groups(3);
  groups[0] = 0;
  groups[1] = 10;
  groups[2] = 20;
  dh::SegmentedArgSort<false>(dh::ToSpan(values), dh::ToSpan(groups),
                              dh::ToSpan(sorted_idx));
  ASSERT_FALSE(thrust::is_sorted(thrust::device, sorted_idx.begin(),
                                 sorted_idx.end(), thrust::greater<size_t>{}));
  ASSERT_TRUE(thrust::is_sorted(sorted_idx.begin(), sorted_idx.begin() + 10,
                                thrust::greater<size_t>{}));
  ASSERT_TRUE(thrust::is_sorted(sorted_idx.begin() + 10, sorted_idx.end(),
                                thrust::greater<size_t>{}));
}

namespace {
// Atomic add as type cast for test.
XGBOOST_DEV_INLINE int64_t atomicAdd(int64_t *dst, int64_t src) {  // NOLINT
  uint64_t* u_dst = reinterpret_cast<uint64_t*>(dst);
  uint64_t u_src = *reinterpret_cast<uint64_t*>(&src);
  uint64_t ret = ::atomicAdd(u_dst, u_src);
  return *reinterpret_cast<int64_t*>(&ret);
}
}

void TestAtomicAdd() {
  size_t n_elements = 1024;
  dh::device_vector<int64_t> result_a(1, 0);
  auto d_result_a = result_a.data().get();

  dh::device_vector<int64_t> result_b(1, 0);
  auto d_result_b = result_b.data().get();

  /**
   * Test for simple inputs
   */
  std::vector<int64_t> h_inputs(n_elements);
  for (size_t i = 0; i < h_inputs.size(); ++i) {
    h_inputs[i] = (i % 2 == 0) ? i : -i;
  }
  dh::device_vector<int64_t> inputs(h_inputs);
  auto d_inputs = inputs.data().get();

  dh::LaunchN(n_elements, [=] __device__(size_t i) {
    dh::AtomicAdd64As32(d_result_a, d_inputs[i]);
    atomicAdd(d_result_b, d_inputs[i]);
  });
  ASSERT_EQ(result_a[0], result_b[0]);

  /**
   * Test for positive values that don't fit into 32 bit integer.
   */
  thrust::fill(inputs.begin(), inputs.end(),
               (std::numeric_limits<uint32_t>::max() / 2));
  thrust::fill(result_a.begin(), result_a.end(), 0);
  thrust::fill(result_b.begin(), result_b.end(), 0);
  dh::LaunchN(n_elements, [=] __device__(size_t i) {
    dh::AtomicAdd64As32(d_result_a, d_inputs[i]);
    atomicAdd(d_result_b, d_inputs[i]);
  });
  ASSERT_EQ(result_a[0], result_b[0]);
  ASSERT_GT(result_a[0], std::numeric_limits<uint32_t>::max());
  CHECK_EQ(thrust::reduce(inputs.begin(), inputs.end(), int64_t(0)), result_a[0]);

  /**
   * Test for negative values that don't fit into 32 bit integer.
   */
  thrust::fill(inputs.begin(), inputs.end(),
               (std::numeric_limits<int32_t>::min() / 2));
  thrust::fill(result_a.begin(), result_a.end(), 0);
  thrust::fill(result_b.begin(), result_b.end(), 0);
  dh::LaunchN(n_elements, [=] __device__(size_t i) {
    dh::AtomicAdd64As32(d_result_a, d_inputs[i]);
    atomicAdd(d_result_b, d_inputs[i]);
  });
  ASSERT_EQ(result_a[0], result_b[0]);
  ASSERT_LT(result_a[0], std::numeric_limits<int32_t>::min());
  CHECK_EQ(thrust::reduce(inputs.begin(), inputs.end(), int64_t(0)), result_a[0]);
}

TEST(AtomicAdd, Int64) {
  TestAtomicAdd();
}
}  // namespace xgboost
