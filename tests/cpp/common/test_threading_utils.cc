/**
 * Copyright 2019-2026, XGBoost Contributors
 */
#include <dmlc/omp.h>  // for omp_get_num_threads, omp_get_thread_num, omp_in_parallel
#include <gtest/gtest.h>

#include <cstddef>  // for std::size_t
#include <memory>   // for make_shared, make_unique, shared_ptr

#include "../../../src/common/threading_utils.h"  // BlockedSpace2d,ParallelFor2d,ParallelFor
#include "xgboost/context.h"                      // Context

namespace xgboost::common {
TEST(ParallelFor2d, CreateBlockedSpace2d) {
  constexpr size_t kDim1 = 5;
  constexpr size_t kDim2 = 3;
  constexpr size_t kGrainSize = 1;

  BlockedSpace2d space(kDim1, [&](size_t) { return kDim2; }, kGrainSize);

  ASSERT_EQ(kDim1 * kDim2, space.Size());

  for (size_t i = 0; i < kDim1; i++) {
    for (size_t j = 0; j < kDim2; j++) {
      ASSERT_EQ(space.GetFirstDimension(i * kDim2 + j), i);
      ASSERT_EQ(j, space.GetRange(i * kDim2 + j).begin());
      ASSERT_EQ(j + kGrainSize, space.GetRange(i * kDim2 + j).end());
    }
  }
}

TEST(ParallelFor2d, Test) {
  constexpr size_t kDim1 = 100;
  constexpr size_t kDim2 = 15;
  constexpr size_t kGrainSize = 2;

  // working space is matrix of size (kDim1 x kDim2)
  std::vector<int> matrix(kDim1 * kDim2, 0);
  BlockedSpace2d space(kDim1, [&](size_t) { return kDim2; }, kGrainSize);
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"nthread", "4"}});
  ASSERT_EQ(ctx.nthread, 4);

  auto const n_threads = ctx.Threads();
  ParallelFor2d(
      space, n_threads,
      WithWorker([&, limit = std::make_unique<int>(n_threads)](size_t i, Range1d r, Worker worker) {
        EXPECT_EQ(worker.Id(), omp_get_thread_num());
        EXPECT_EQ(worker.Count(), omp_get_num_threads());
        EXPECT_LE(worker.Count(), *limit);
        for (auto j = r.begin(); j < r.end(); ++j) {
          matrix[i * kDim2 + j] += 1;
        }
      }));

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
  std::vector<size_t> dim2{1024, 500, 255, 5, 10000};
  BlockedSpace2d space(kDim1, [&](size_t i) { return dim2[i]; }, kGrainSize);

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
  ParallelFor(n, 1, [&](auto) { ASSERT_FALSE(omp_in_parallel()); });
}

TEST(ParallelFor, WithWorker) {
  for (auto n_threads : {1, 4}) {
    // Owning the thread limit makes the callback move-only.
    auto make_callback = [n_threads] {
      return WithWorker([limit = std::make_unique<int>(n_threads)](auto, Worker worker) {
        EXPECT_EQ(worker.Id(), omp_get_thread_num());
        EXPECT_EQ(worker.Count(), omp_get_num_threads());
        EXPECT_GE(worker.Id(), 0);
        EXPECT_LT(worker.Id(), worker.Count());
        EXPECT_LE(worker.Count(), *limit);
      });
    };
    ParallelFor(17, n_threads, make_callback());
    ParallelFor(17, n_threads, Sched::Dyn(2), make_callback());
    ParallelFor1d<3>(17, n_threads, make_callback());
    ParallelForBlock(17, n_threads, make_callback());
  }
  // Nested serial loops report worker {0, 1}, not the outer thread ID and team size.
  ParallelFor(4, 4, [](auto) {
    ParallelFor(1, 1, WithWorker([](auto, Worker inner) {
                  EXPECT_EQ(inner.Id(), 0);
                  EXPECT_EQ(inner.Count(), 1);
                }));
  });
}

TEST(OmpGetNumThreads, Max) {
#if defined(_OPENMP)
  auto n_threads = OmpGetNumThreads(1 << 18);
  ASSERT_LE(n_threads, std::thread::hardware_concurrency());  // le due to container
  n_threads = OmpGetNumThreads(0);
  ASSERT_GE(n_threads, 1);
  ASSERT_LE(n_threads, std::thread::hardware_concurrency());
#endif
}

TEST(MemStackAllocator, ObjectLifetime) {
  auto value = std::make_shared<int>(7);
  for (std::size_t n : {0, 2, 3}) {
    {
      MemStackAllocator<std::shared_ptr<int>, 2> storage(n, value);
      ASSERT_EQ(value.use_count(), n + 1);
      for (std::size_t i = 0; i < n; ++i) {
        ASSERT_EQ(storage[i], value);
      }
    }
    ASSERT_EQ(value.use_count(), 1);
  }
}
}  // namespace xgboost::common
