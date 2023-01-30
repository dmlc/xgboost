/**
 * Copyright 2019-2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <cstddef>  // std::size_t

#include "../../../src/common/threading_utils.h"  // BlockedSpace2d,ParallelFor2d,ParallelFor
#include "dmlc/omp.h"                             // omp_in_parallel
#include "xgboost/context.h"                      // Context

namespace xgboost {
namespace common {

TEST(ParallelFor2d, CreateBlockedSpace2d) {
  constexpr size_t kDim1 = 5;
  constexpr size_t kDim2 = 3;
  constexpr size_t kGrainSize = 1;

  BlockedSpace2d space(
      kDim1, [&](size_t) { return kDim2; }, kGrainSize);

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
  BlockedSpace2d space(
      kDim1, [&](size_t) { return kDim2; }, kGrainSize);
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"nthread", "4"}});
  ASSERT_EQ(ctx.nthread, 4);

  ParallelFor2d(space, ctx.Threads(), [&](size_t i, Range1d r) {
    for (auto j = r.begin(); j < r.end(); ++j) {
      matrix[i * kDim2 + j] += 1;
    }
  });

  for (size_t i = 0; i < kDim1 * kDim2; i++) {
    ASSERT_EQ(matrix[i], 1);
  }
}

TEST(ParallelFor2d, NonUniform) {
  constexpr size_t kDim1 = 5;
  constexpr size_t kGrainSize = 256;

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

  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"nthread", "4"}});
  ASSERT_EQ(ctx.nthread, 4);

  ParallelFor2d(space, ctx.Threads(), [&](size_t i, Range1d r) {
    for (auto j = r.begin(); j < r.end(); ++j) {
      working_space[i][j] += 1;
    }
  });

  for (size_t i = 0; i < kDim1; i++) {
    for (size_t j = 0; j < dim2[i]; j++) {
      ASSERT_EQ(working_space[i][j], 1);
    }
  }
}

TEST(ParallelFor, Basic) {
  Context ctx;
  std::size_t n{16};
  auto n_threads = ctx.Threads();
  ParallelFor(n, n_threads, [&](auto i) {
    ASSERT_EQ(ctx.Threads(), 1);
    if (n_threads > 1) {
      ASSERT_TRUE(omp_in_parallel());
    }
    ASSERT_LT(i, n);
  });
  ASSERT_FALSE(omp_in_parallel());
}
}  // namespace common
}  // namespace xgboost
