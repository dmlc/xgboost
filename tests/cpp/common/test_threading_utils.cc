/**
 * Copyright 2019-2026, XGBoost Contributors
 */
#include <dmlc/omp.h>  // for omp_get_thread_num, omp_set_num_threads, kmp_set_blocktime
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
#if defined(_OPENMP)
namespace {
// Whether the runtime grants a team of n threads, so team sizes and static assignments can be
// checked exactly. OMP_DYNAMIC or a lower OMP_THREAD_LIMIT may shrink the team.
bool FullTeam(int n) {
  if (omp_get_dynamic()) {
    return false;
  }
#if !defined(_MSC_VER)
  // MSVC doesn't implement omp_get_thread_limit.
  if (omp_get_thread_limit() < n) {
    return false;
  }
#endif
  return true;
}
}  // namespace
#endif
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

TEST(ParallelFor2d, TeamSize) {
  // A single block runs on the calling thread. Otherwise the OpenMP team keeps all n_threads
  // when OpenMPExecutor::FixedTeam(), or as many threads as blocks if not; block i goes to
  // worker i.
  for (std::size_t n_blocks : {1, 2, 3}) {
    BlockedSpace2d space(n_blocks, [](std::size_t) { return std::size_t{1}; }, 1);
    std::array<std::atomic<int>, 3> visits{};
#if defined(_OPENMP)
    // Read right before the call, as the executor does: with libomp the answer can change at run
    // time (see OpenMPExecutor::FixedTeam).
    bool const fixed = OpenMPExecutor::FixedTeam();
#endif
    ParallelFor2d(space, 4, WithWorker([&](std::size_t i, Range1d, Worker worker) {
                    ++visits[i];
                    EXPECT_LT(worker.Id(), 4);
                    if (n_blocks == 1) {
                      EXPECT_EQ(worker.Id(), 0);
                    }
#if defined(_OPENMP)
                    if (n_blocks == 1) {
                      EXPECT_FALSE(omp_in_parallel());
                    } else if (FullTeam(4)) {
                      EXPECT_EQ(worker.Id(), static_cast<std::int32_t>(i));
                      EXPECT_EQ(omp_get_num_threads(), fixed ? 4 : static_cast<int>(n_blocks));
                    }
#endif
                  }));
    for (std::size_t i = 0; i < visits.size(); ++i) {
      EXPECT_EQ(visits[i].load(), i < n_blocks ? 1 : 0);
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
    for (int size : {0, 1, 2, 3, 19}) {
      std::array<std::atomic<int>, 19> visits{};
      ParallelFor(size, 4, sched, [&](auto i) { ++visits[i]; });
      for (std::size_t i = 0; i < visits.size(); ++i) {
        EXPECT_EQ(visits[i].load(), i < static_cast<std::size_t>(size) ? 1 : 0);
      }
    }
  }
}

TEST(ParallelFor, TeamSize) {
  // A single iteration runs on the calling thread. Otherwise the OpenMP team keeps all
  // n_threads when OpenMPExecutor::FixedTeam(), or is capped at the loop size if not.
  for (auto sched : {Sched::Auto(), Sched::Static(), Sched::Dyn(), Sched::Guided()}) {
    for (int size : {1, 2, 3}) {
#if defined(_OPENMP)
      bool const fixed = OpenMPExecutor::FixedTeam();
#endif
      ParallelFor(size, 4, sched, WithWorker([&](auto i, Worker worker) {
                    EXPECT_LT(worker.Id(), 4);
                    if (size == 1) {
                      EXPECT_EQ(worker.Id(), 0);
                    }
#if defined(_OPENMP)
                    if (size == 1) {
                      EXPECT_FALSE(omp_in_parallel());
                    } else if (FullTeam(4)) {
                      auto team = fixed ? 4 : size;
                      EXPECT_LT(worker.Id(), team);
                      EXPECT_EQ(omp_get_num_threads(), team);
                      if (sched.sched == Sched::kStatic) {
                        // A static schedule still gives iteration i to worker i.
                        EXPECT_EQ(worker.Id(), i);
                      }
                    }
#endif
                  }));
    }
  }
}

TEST(ParallelFor, FixedTeamWithLibgomp) {
#if defined(_OPENMP) && defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && \
    !defined(__NVCOMPILER)
  // GCC builds use libgomp, which needs a fixed team.
  EXPECT_TRUE(OpenMPExecutor::kFixedTeam);
  EXPECT_TRUE(OpenMPExecutor::FixedTeam());
#else
  GTEST_SKIP() << "Not a GCC OpenMP build.";
#endif
}

TEST(ParallelFor, FixedTeamFollowsBlocktime) {
#if defined(XGBOOST_KMP_BLOCKTIME_API)
  // With libomp, the team stays at the requested size unless idle threads sleep at once. The
  // checks run on a new thread, a separate OpenMP root whose settings are discarded when it ends,
  // so the blocktime set here does not leak into other tests.
  bool ignored{false};
  std::thread{[&] {
    for (int blocktime : {0, 1}) {
      kmp_set_blocktime(blocktime);
      if (kmp_get_blocktime() != blocktime) {
        // An infinite blocktime (KMP_BLOCKTIME=infinite, OMP_WAIT_POLICY=active) can't be changed.
        ignored = true;
        return;
      }
      bool const fixed = blocktime > 0;
      EXPECT_EQ(OpenMPExecutor::FixedTeam(), fixed);
      if (!FullTeam(4)) {
        continue;
      }
      std::atomic<int> team{0};
      ParallelFor(3, 4, [&](int) { team = omp_get_num_threads(); });
      EXPECT_EQ(team.load(), fixed ? 4 : 3);
      BlockedSpace2d space(3, [](std::size_t) { return std::size_t{1}; }, 1);
      ParallelFor2d(space, 4, [&](std::size_t, Range1d) { team = omp_get_num_threads(); });
      EXPECT_EQ(team.load(), fixed ? 4 : 3);
    }
  }}.join();
  if (ignored) {
    GTEST_SKIP() << "The OpenMP runtime ignored kmp_set_blocktime.";
  }
#else
  GTEST_SKIP() << "Requires an OpenMP runtime with kmp_get_blocktime.";
#endif
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

TEST(ParallelFor, SingleIterationException) {
  // A single iteration or block runs on the calling thread and still propagates exceptions.
  EXPECT_THROW(ParallelFor(1, 4, [](auto) { throw std::runtime_error{"task failed"}; }),
               std::runtime_error);
  BlockedSpace2d space(1, [](std::size_t) { return std::size_t{1}; }, 1);
  EXPECT_THROW(ParallelFor2d(space, 4,
                             [](std::size_t, Range1d) { throw std::runtime_error{"task failed"}; }),
               std::runtime_error);
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
