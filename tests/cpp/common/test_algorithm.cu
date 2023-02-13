/**
 * Copyright 2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <thrust/copy.h>      // copy
#include <thrust/sequence.h>  // sequence
#include <thrust/sort.h>      // is_sorted

#include <algorithm>          // is_sorted
#include <cstddef>            // size_t

#include "../../../src/common/algorithm.cuh"
#include "../../../src/common/device_helpers.cuh"
#include "../helpers.h"  // CreateEmptyGenericParam

namespace xgboost {
namespace common {
void TestSegmentedArgSort() {
  Context ctx;
  ctx.gpu_id = 0;

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
    auto group_sorted_idx = common::Span<size_t>(h_sorted_index)
                                .subspan(offset_ptr[i - 1], offset_ptr[i] - offset_ptr[i - 1]);
    ASSERT_TRUE(std::is_sorted(group_sorted_idx.begin(), group_sorted_idx.end(), std::greater<>{}));
    ASSERT_EQ(group_sorted_idx.back(), 0);
    for (auto j : group_sorted_idx) {
      ASSERT_LT(j, group_sorted_idx.size());
    }
  }
}

TEST(Algorithm, SegmentedArgSort) { TestSegmentedArgSort(); }

TEST(Algorithm, GpuArgSort) {
  Context ctx;
  ctx.gpu_id = 0;

  dh::device_vector<float> values(20);
  dh::Iota(dh::ToSpan(values));                                    // accending
  dh::device_vector<size_t> sorted_idx(20);
  dh::ArgSort<false>(dh::ToSpan(values), dh::ToSpan(sorted_idx));  // sort to descending
  ASSERT_TRUE(thrust::is_sorted(thrust::device, sorted_idx.begin(), sorted_idx.end(),
                                thrust::greater<size_t>{}));

  dh::Iota(dh::ToSpan(values));
  dh::device_vector<size_t> groups(3);
  groups[0] = 0;
  groups[1] = 10;
  groups[2] = 20;
  SegmentedArgSort<false, false>(&ctx, dh::ToSpan(values), dh::ToSpan(groups),
                                 dh::ToSpan(sorted_idx));
  ASSERT_FALSE(thrust::is_sorted(thrust::device, sorted_idx.begin(), sorted_idx.end(),
                                 thrust::greater<size_t>{}));
  ASSERT_TRUE(
      thrust::is_sorted(sorted_idx.begin(), sorted_idx.begin() + 10, thrust::greater<size_t>{}));
  ASSERT_TRUE(
      thrust::is_sorted(sorted_idx.begin() + 10, sorted_idx.end(), thrust::greater<size_t>{}));
}

TEST(Algorithm, SegmentedSequence) {
  dh::device_vector<std::size_t> idx(16);
  dh::device_vector<std::size_t> ptr(3);
  Context ctx = CreateEmptyGenericParam(0);
  ptr[0] = 0;
  ptr[1] = 4;
  ptr[2] = idx.size();
  SegmentedSequence(&ctx, dh::ToSpan(ptr), dh::ToSpan(idx));
  ASSERT_EQ(idx[0], 0);
  ASSERT_EQ(idx[4], 0);
  ASSERT_EQ(idx[3], 3);
  ASSERT_EQ(idx[15], 11);
}
}  // namespace common
}  // namespace xgboost
