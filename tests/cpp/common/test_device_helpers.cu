
/*!
 * Copyright 2017 XGBoost contributors
 */
#include <thrust/device_vector.h>
#include <xgboost/base.h>
#include "../../../src/common/device_helpers.cuh"
#include "../helpers.h"
#include "gtest/gtest.h"

using xgboost::common::Span;

void CreateTestData(xgboost::bst_uint num_rows, int max_row_size,
                    thrust::host_vector<int> *row_ptr,
                    thrust::host_vector<xgboost::bst_uint> *rows) {
  row_ptr->resize(num_rows + 1);
  int sum = 0;
  for (xgboost::bst_uint i = 0; i <= num_rows; i++) {
    (*row_ptr)[i] = sum;
    sum += rand() % max_row_size;  // NOLINT

    if (i < num_rows) {
      for (int j = (*row_ptr)[i]; j < sum; j++) {
        (*rows).push_back(i);
      }
    }
  }
}

TEST(SumReduce, Test) {
  thrust::device_vector<float> data(100, 1.0f);
  dh::CubMemory temp;
  auto sum = dh::SumReduction(&temp, data.data().get(), data.size());
  ASSERT_NEAR(sum, 100.0f, 1e-5);
}

template <typename T, typename Comp = thrust::less<T>>
void TestUpperBoundImpl(const std::vector<T> &vec, T val_to_find,
                        const Comp &comp = Comp()) {
  EXPECT_EQ(dh::UpperBound(vec.data(), vec.size(), val_to_find, comp),
            std::upper_bound(vec.begin(), vec.end(), val_to_find, comp) - vec.begin());
}

template <typename T, typename Comp = thrust::less<T>>
void TestLowerBoundImpl(const std::vector<T> &vec, T val_to_find,
                        const Comp &comp = Comp()) {
  EXPECT_EQ(dh::LowerBound(vec.data(), vec.size(), val_to_find, comp),
            std::lower_bound(vec.begin(), vec.end(), val_to_find, comp) - vec.begin());
}

TEST(UpperBound, DataAscending) {
  std::vector<int> hvec{0, 3, 5, 5, 7, 8, 9, 10, 10};

  // Test boundary conditions
  TestUpperBoundImpl(hvec, hvec.front());  // Result 1
  TestUpperBoundImpl(hvec, hvec.front() - 1);  // Result 0
  TestUpperBoundImpl(hvec, hvec.back() + 1);  // Result hvec.size()
  TestUpperBoundImpl(hvec, hvec.back());  // Result hvec.size()

  // Test other values - both missing and present
  TestUpperBoundImpl(hvec, 3);  // Result 2
  TestUpperBoundImpl(hvec, 4);  // Result 2
  TestUpperBoundImpl(hvec, 5);  // Result 4
}

TEST(UpperBound, DataDescending) {
  std::vector<int> hvec{10, 10, 9, 8, 7, 5, 5, 3, 0, 0};
  const auto &comparator = thrust::greater<int>();

  // Test boundary conditions
  TestUpperBoundImpl(hvec, hvec.front(), comparator);  // Result 2
  TestUpperBoundImpl(hvec, hvec.front() + 1, comparator);  // Result 0
  TestUpperBoundImpl(hvec, hvec.back(), comparator);  // Result hvec.size()
  TestUpperBoundImpl(hvec, hvec.back() - 1, comparator);  // Result hvec.size()

  // Test other values - both missing and present
  TestUpperBoundImpl(hvec, 9, comparator);  // Result 3
  TestUpperBoundImpl(hvec, 7, comparator);  // Result 5
  TestUpperBoundImpl(hvec, 4, comparator);  // Result 7
  TestUpperBoundImpl(hvec, 8, comparator);  // Result 4
}

TEST(LowerBound, DataAscending) {
  std::vector<int> hvec{0, 3, 5, 5, 7, 8, 9, 10, 10};

  // Test boundary conditions
  TestLowerBoundImpl(hvec, hvec.front());  // Result 0
  TestLowerBoundImpl(hvec, hvec.front() - 1);  // Result 0
  TestLowerBoundImpl(hvec, hvec.back());  // Result 7
  TestLowerBoundImpl(hvec, hvec.back() + 1);  // Result hvec.size()

  // Test other values - both missing and present
  TestLowerBoundImpl(hvec, 3);  // Result 1
  TestLowerBoundImpl(hvec, 4);  // Result 2
  TestLowerBoundImpl(hvec, 5);  // Result 2
}

TEST(LowerBound, DataDescending) {
  std::vector<int> hvec{10, 10, 9, 8, 7, 5, 5, 3, 0, 0};
  const auto &comparator = thrust::greater<int>();

  // Test boundary conditions
  TestLowerBoundImpl(hvec, hvec.front(), comparator);  // Result 0
  TestLowerBoundImpl(hvec, hvec.front() + 1, comparator);  // Result 0
  TestLowerBoundImpl(hvec, hvec.back(), comparator);  // Result 8
  TestLowerBoundImpl(hvec, hvec.back() - 1, comparator);  // Result hvec.size()

  // Test other values - both missing and present
  TestLowerBoundImpl(hvec, 9, comparator);  // Result 2
  TestLowerBoundImpl(hvec, 7, comparator);  // Result 4
  TestLowerBoundImpl(hvec, 4, comparator);  // Result 7
  TestLowerBoundImpl(hvec, 8, comparator);  // Result 3
}
