#include <cstddef>
#include <gtest/gtest.h>

#include "../../../src/common/column_matrix.h"
#include "../../../src/common/threading_utils.h"

namespace xgboost {
namespace common {

TEST(CreateBlockedSpace2d, Test) {
  constexpr size_t kDim1 = 5;
  constexpr size_t kDim2 = 3;
  constexpr size_t kGrainSize = 1;

  BlockedSpace2d space(kDim1, [&](size_t i) {
      return kDim2;
  }, kGrainSize);

  ASSERT_EQ(kDim1 * kDim2, space.Size());

  for (size_t i = 0; i < kDim1; i++) {
    for (size_t j = 0; j < kDim2; j++) {
      ASSERT_EQ(space.GetFirstDimension(i*kDim2 + j), i);
      ASSERT_EQ(j, space.GetRange(i*kDim2 + j).begin());
      ASSERT_EQ(j + kGrainSize, space.GetRange(i*kDim2 + j).end());
    }
  }
}

TEST(ParallelFor2d, Test) {
  constexpr size_t kDim1 = 100;
  constexpr size_t kDim2 = 15;
  constexpr size_t kGrainSize = 2;

  // working space is matrix of size (kDim1 x kDim2)
  std::vector<int> matrix(kDim1 * kDim2, 0);
  BlockedSpace2d space(kDim1, [&](size_t i) {
      return kDim2;
  }, kGrainSize);

  auto old = omp_get_max_threads();
  omp_set_num_threads(4);

  ParallelFor2d(space, omp_get_max_threads(), [&](size_t i, Range1d r) {
    for (auto j = r.begin(); j < r.end(); ++j) {
      matrix[i*kDim2 + j] += 1;
    }
  });

  for (size_t i = 0; i < kDim1 * kDim2; i++) {
    ASSERT_EQ(matrix[i], 1);
  }

  omp_set_num_threads(old);
}

TEST(ParallelFor2dNonUniform, Test) {
  constexpr size_t kDim1 = 5;
  constexpr size_t kGrainSize = 256;

  auto old = omp_get_max_threads();
  omp_set_num_threads(4);

  // here are quite non-uniform distribution in space
  // but ParallelFor2d should split them by blocks with max size = kGrainSize
  // and process in balanced manner (optimal performance)
  std::vector<size_t> dim2 { 1024, 500, 255, 5, 10000 };
  BlockedSpace2d space(kDim1, [&](size_t i) {
      return dim2[i];
  }, kGrainSize);

  std::vector<std::vector<int>> working_space(kDim1);
  for (size_t i = 0; i < kDim1; i++) {
    working_space[i].resize(dim2[i], 0);
  }

  ParallelFor2d(space, omp_get_max_threads(), [&](size_t i, Range1d r) {
    for (auto j = r.begin(); j < r.end(); ++j) {
      working_space[i][j] += 1;
    }
  });

  for (size_t i = 0; i < kDim1; i++) {
    for (size_t j = 0; j < dim2[i]; j++) {
      ASSERT_EQ(working_space[i][j], 1);
    }
  }

  omp_set_num_threads(old);
}

}  // namespace common
}  // namespace xgboost
