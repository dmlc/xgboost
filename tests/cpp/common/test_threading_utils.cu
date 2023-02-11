/**
 * Copyright 2021-2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <thrust/copy.h>  // thrust::copy

#include "../../../src/common/device_helpers.cuh"
#include "../../../src/common/threading_utils.cuh"

namespace xgboost {
namespace common {
TEST(SegmentedTrapezoidThreads, Basic) {
  size_t constexpr kElements = 24, kGroups = 3;
  dh::device_vector<size_t> offset_ptr(kGroups + 1, 0);
  offset_ptr[0] = 0;
  offset_ptr[1] = 8;
  offset_ptr[2] = 16;
  offset_ptr[kGroups] = kElements;

  size_t h = 1;
  dh::device_vector<size_t> thread_ptr(kGroups + 1, 0);
  size_t total = SegmentedTrapezoidThreads(dh::ToSpan(offset_ptr), dh::ToSpan(thread_ptr), h);
  ASSERT_EQ(total, kElements - kGroups);

  h = 2;
  SegmentedTrapezoidThreads(dh::ToSpan(offset_ptr), dh::ToSpan(thread_ptr), h);
  std::vector<size_t> h_thread_ptr(thread_ptr.size());
  thrust::copy(thread_ptr.cbegin(), thread_ptr.cend(), h_thread_ptr.begin());
  for (size_t i = 1; i < h_thread_ptr.size(); ++i) {
    ASSERT_EQ(h_thread_ptr[i] - h_thread_ptr[i - 1], 13);
  }

  h = 7;
  SegmentedTrapezoidThreads(dh::ToSpan(offset_ptr), dh::ToSpan(thread_ptr), h);
  thrust::copy(thread_ptr.cbegin(), thread_ptr.cend(), h_thread_ptr.begin());
  for (size_t i = 1; i < h_thread_ptr.size(); ++i) {
    ASSERT_EQ(h_thread_ptr[i] - h_thread_ptr[i - 1], 28);
  }
}

TEST(SegmentedTrapezoidThreads, Unravel) {
  size_t i = 0, j = 0;
  size_t constexpr kN = 8;

  UnravelTrapeziodIdx(6, kN, &i, &j);
  ASSERT_EQ(i, 0);
  ASSERT_EQ(j, 7);

  UnravelTrapeziodIdx(12, kN, &i, &j);
  ASSERT_EQ(i, 1);
  ASSERT_EQ(j, 7);

  UnravelTrapeziodIdx(15, kN, &i, &j);
  ASSERT_EQ(i, 2);
  ASSERT_EQ(j, 5);

  UnravelTrapeziodIdx(21, kN, &i, &j);
  ASSERT_EQ(i, 3);
  ASSERT_EQ(j, 7);

  UnravelTrapeziodIdx(25, kN, &i, &j);
  ASSERT_EQ(i, 5);
  ASSERT_EQ(j, 6);

  UnravelTrapeziodIdx(27, kN, &i, &j);
  ASSERT_EQ(i, 6);
  ASSERT_EQ(j, 7);
}
}  // namespace common
}  // namespace xgboost
