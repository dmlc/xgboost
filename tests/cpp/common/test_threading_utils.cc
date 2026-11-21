/**
 * Copyright 2019-2026, XGBoost Contributors
 */
#include <dmlc/omp.h>  // for omp_get_thread_num, omp_get_max_threads, omp_set_num_threads
#include <gtest/gtest.h>

#include <array>      // for array
#include <atomic>     // for atomic
#include <chrono>     // for milliseconds
#include <cstddef>    // for size_t
#include <memory>     // for make_shared, make_unique, shared_ptr
#include <stdexcept>  // for runtime_error
#include <thread>     // for sleep_for, thread
#include <vector>     // for vector

#include "../../../src/common/threading_utils.h"  // BlockedSpace2d,ParallelFor2d,ParallelFor
#include "xgboost/global_config.h"                // for GlobalConfigThreadLocalStore

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

TEST(ParallelFor2d, Coverage) {
  std::vector<std::size_t> sizes{0, 1, 5, 8, 19};
  BlockedSpace2d space(sizes.size(), [&](auto i) { return sizes[i]; }, 3);
  for (auto n_threads : {1, 4}) {
    std::vector<std::vector<int>> visits;
    for (auto size : sizes) {
      visits.emplace_back(size, 0);
    }
    auto visit = [&](std::size_t i, Range1d range) {
      for (auto j = range.begin(); j < range.end(); ++j) {
        ++visits[i][j];
      }
    };
    if (n_threads == 1) {
      ParallelFor2d(space, n_threads, visit);
    } else {
      ParallelFor2d(space, n_threads,
                    WithWorker([&, limit = std::make_unique<int>(n_threads)](
                                   std::size_t i, Range1d range, Worker worker) {
                      ASSERT_GE(worker.Id(), 0);
                      ASSERT_LT(worker.Id(), *limit);
                      visit(i, range);
                    }));
    }
    for (auto const& row : visits) {
      for (auto count : row) {
        EXPECT_EQ(count, 1);
      }
    }
  }
}

TEST(ParallelFor, WithWorker) {
  for (auto n_threads : {1, 4}) {
    // Owning the thread limit makes the callback move-only.
    auto make_callback = [n_threads] {
      return WithWorker([limit = std::make_unique<int>(n_threads)](auto, Worker worker) {
#if defined(_OPENMP)
        EXPECT_EQ(worker.Id(), omp_get_thread_num());
#endif
        EXPECT_GE(worker.Id(), 0);
        EXPECT_LT(worker.Id(), *limit);
      });
    };
    ParallelFor(std::size_t{17}, n_threads, make_callback());
    ParallelFor1d<3>(17, n_threads, make_callback());
    ParallelForBlock(17, n_threads, make_callback());
  }
  // Nested serial loops report worker ID 0, not the outer thread ID.
  ParallelFor(4, 4, [](auto) {
    ParallelFor(1, 1, WithWorker([](auto, Worker inner) { EXPECT_EQ(inner.Id(), 0); }));
  });
}

TEST(ParallelFor, Schedules) {
  for (auto sched : {Sched::Auto(), Sched::Static(), Sched::Static(2), Sched::Dyn(), Sched::Dyn(2),
                     Sched::Guided()}) {
    for (int size : {0, 1, 19}) {
      std::array<std::atomic<int>, 19> visits{};
      ParallelFor(size, 4, sched, [&](auto i) { ++visits[i]; });
      for (std::size_t i = 0; i < visits.size(); ++i) {
        EXPECT_EQ(visits[i].load(), i < static_cast<std::size_t>(size) ? 1 : 0);
      }
    }
  }
}

TEST(ParallelFor, ExceptionDrainsTasks) {
  bool finished{false};
  EXPECT_THROW(ParallelFor(2, 2,
                           [&](auto i) {
                             if (i == 0) {
                               throw std::runtime_error{"task failed"};
                             }
                             std::this_thread::sleep_for(std::chrono::milliseconds{10});
                             finished = true;
                           }),
               std::runtime_error);
  EXPECT_TRUE(finished);
}

TEST(GetNumThreads, ConfiguredLimit) {
#if defined(_OPENMP)
  auto previous = omp_get_max_threads();
  omp_set_num_threads(2);
#else
  auto* config = GlobalConfigThreadLocalStore::Get();
  auto previous = config->nthread;
  config->nthread = 2;
#endif
  EXPECT_LE(GetNumThreads(0), 2);
  EXPECT_EQ(GetNumThreads(8), GetNumThreads(0));
  EXPECT_EQ(GetNumThreads(1), 1);
#if defined(_OPENMP)
  ParallelFor(2, 2, [](auto) { EXPECT_EQ(GetNumThreads(0), 1); });
  omp_set_num_threads(previous);
#else
  config->nthread = previous;
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
