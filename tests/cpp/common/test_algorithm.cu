/**
 * Copyright 2023-2025, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <thrust/copy.h>      // copy
#include <thrust/sequence.h>  // sequence
#include <thrust/sort.h>      // is_sorted

#include <algorithm>            // is_sorted
#include <cstddef>              // size_t
#include <cuda/std/functional>  // for greater

#include "../../../src/common/algorithm.cuh"
#include "../../../src/common/device_helpers.cuh"
#include "../helpers.h"  // MakeCUDACtx

namespace xgboost::common {
void TestSegmentedArgSort() {
  auto ctx = MakeCUDACtx(0);

  size_t constexpr kElements = 100, kGroups = 3;
  dh::device_vector<size_t> sorted_idx(kElements, 0);
  dh::device_vector<size_t> offset_ptr(kGroups + 1, 0);
  offset_ptr[0] = 0;
  offset_ptr[1] = 2;
  offset_ptr[2] = 78;
  offset_ptr[kGroups] = kElements;
  auto d_offset_ptr = dh::ToSpan(offset_ptr);

  auto d_sorted_idx = dh::ToSpan(sorted_idx);
  dh::LaunchN(sorted_idx.size(), [=] XGBOOST_DEVICE(size_t idx) {
    auto group = dh::SegmentId(d_offset_ptr, idx);
    d_sorted_idx[idx] = idx - d_offset_ptr[group];
  });

  dh::device_vector<float> values(kElements, 0.0f);
  thrust::sequence(values.begin(), values.end(), 0.0f);
  SegmentedArgSort<false, true>(&ctx, dh::ToSpan(values), d_offset_ptr, d_sorted_idx);

  std::vector<size_t> h_sorted_index(sorted_idx.size());
  thrust::copy(sorted_idx.begin(), sorted_idx.end(), h_sorted_index.begin());

  for (size_t i = 1; i < kGroups + 1; ++i) {
    auto group_sorted_idx =
        Span<size_t>(h_sorted_index).subspan(offset_ptr[i - 1], offset_ptr[i] - offset_ptr[i - 1]);
    ASSERT_TRUE(std::is_sorted(group_sorted_idx.begin(), group_sorted_idx.end(), std::greater<>{}));
    ASSERT_EQ(group_sorted_idx.back(), 0);
    for (auto j : group_sorted_idx) {
      ASSERT_LT(j, group_sorted_idx.size());
    }
  }
}

TEST(Algorithm, SegmentedArgSort) { TestSegmentedArgSort(); }

TEST(Algorithm, SegmentedSortKeysInPlace) {
  auto ctx = MakeCUDACtx(0);
  // Include an empty segment and an untouched prefix and suffix.
  std::vector<int> input{99, 4, 1, 3, 2, 8, 6, 7, 88};
  std::vector<std::size_t> offsets{1, 5, 5, 8};
  dh::device_vector<std::size_t> groups(offsets);
  Span<std::size_t const> group_ptr{groups.data().get(), groups.size()};
  for (bool descending : {false, true}) {
    dh::device_vector<int> values(input);
    if (descending) {
      SegmentedSortKeys<true>(&ctx, group_ptr, dh::ToSpan(values));
    } else {
      SegmentedSortKeys<false>(&ctx, group_ptr, dh::ToSpan(values));
    }
    std::vector<int> result(values.size());
    thrust::copy(values.begin(), values.end(), result.begin());
    auto expected = descending ? std::vector<int>{99, 4, 3, 2, 1, 8, 7, 6, 88}
                               : std::vector<int>{99, 1, 2, 3, 4, 6, 7, 8, 88};
    EXPECT_EQ(result, expected);
  }
}

TEST(Algorithm, InclusiveScanInPlace) {
  auto ctx = MakeCUDACtx(0);
  std::vector<int> input{1, 2, 3, 4};
  dh::device_vector<int> values(input);
  InclusiveScan(&ctx, values.data().get(), values.data().get(), std::plus<int>{},
                std::size_t{input.size()});
  std::vector<int> result(values.size());
  thrust::copy(values.begin(), values.end(), result.begin());
  EXPECT_EQ(result, (std::vector<int>{1, 3, 6, 10}));
}

TEST(Algorithm, GpuArgSort) {
  auto ctx = MakeCUDACtx(0);

  dh::device_vector<float> values(20);
  dh::Iota(dh::ToSpan(values), ctx.CUDACtx()->Stream());  // accending
  dh::device_vector<size_t> sorted_idx(20);
  ArgSort<false>(&ctx, dh::ToSpan(values), dh::ToSpan(sorted_idx));  // sort to descending
  ASSERT_TRUE(thrust::is_sorted(ctx.CUDACtx()->CTP(), sorted_idx.begin(), sorted_idx.end(),
                                cuda::std::greater<size_t>{}));

  dh::Iota(dh::ToSpan(values), ctx.CUDACtx()->Stream());
  dh::device_vector<size_t> groups(3);
  groups[0] = 0;
  groups[1] = 10;
  groups[2] = 20;
  SegmentedArgSort<false, false>(&ctx, dh::ToSpan(values), dh::ToSpan(groups),
                                 dh::ToSpan(sorted_idx));
  ASSERT_FALSE(thrust::is_sorted(thrust::device, sorted_idx.begin(), sorted_idx.end(),
                                 cuda::std::greater<size_t>{}));
  ASSERT_TRUE(
      thrust::is_sorted(sorted_idx.begin(), sorted_idx.begin() + 10, cuda::std::greater<size_t>{}));
  ASSERT_TRUE(
      thrust::is_sorted(sorted_idx.begin() + 10, sorted_idx.end(), cuda::std::greater<size_t>{}));
}

TEST(Algorithm, SegmentedSequence) {
  dh::device_vector<std::size_t> idx(16);
  dh::device_vector<std::size_t> ptr(3);
  Context ctx = MakeCUDACtx(0);
  ptr[0] = 0;
  ptr[1] = 4;
  ptr[2] = idx.size();
  SegmentedSequence(&ctx, dh::ToSpan(ptr), dh::ToSpan(idx));
  ASSERT_EQ(idx[0], 0);
  ASSERT_EQ(idx[4], 0);
  ASSERT_EQ(idx[3], 3);
  ASSERT_EQ(idx[15], 11);
}

namespace {
void TestAllOf(std::size_t n) {
  auto ctx = MakeCUDACtx(0);
  dh::device_vector<double> values(n);
  dh::Iota(dh::ToSpan(values), ctx.CUDACtx()->Stream());
  EXPECT_TRUE(AllOf(ctx.CUDACtx()->CTP(), values.cbegin(), values.cend(),
                    [n] XGBOOST_DEVICE(double v) { return v < n; }));
  if (n == 0) {
    return;
  }
  EXPECT_FALSE(AllOf(ctx.CUDACtx()->CTP(), values.cbegin(), values.cend(),
                     [n] XGBOOST_DEVICE(double v) { return v < n && v > 0; }));
}
}  // namespace

TEST(Algorithm, AllOf) {
  TestAllOf(0);
  TestAllOf(1);
  TestAllOf(2);
  TestAllOf(4096);
}
}  // namespace xgboost::common
