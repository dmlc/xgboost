/**
 * Copyright 2024-2026, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/global_config.h>  // for GlobalConfigThreadLocalStore

#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t
#include <future>     // for async, future, launch
#include <stdexcept>  // for runtime_error
#include <thread>     // for this_thread
#include <vector>     // for vector

#include "../../../src/common/threadpool.h"

namespace xgboost::common {
TEST(ThreadPool, NestedExecutor) {
  using Exec = GlobalThreadPoolExecutor;
  auto const n_threads = Exec::MaxNumWorkers();
  std::vector<int> counts(2);
  Exec::Execute(2, 2, Sched::Static(), [&](int i) {
    ASSERT_EQ(Exec::MaxNumWorkers(), 1);
    auto const worker = std::this_thread::get_id();
    // Explicit thread counts must also run inline inside an executor task.
    Exec::Execute(3, 2, Sched::Static(), WithWorker{[&](int, Worker inner) {
                    EXPECT_EQ(std::this_thread::get_id(), worker);
                    EXPECT_EQ(inner.Id(), 0);
                    ++counts[i];
                  }});
    EXPECT_THROW(Exec::Execute(2, 2, Sched::Static(),
                               [](int) { throw std::runtime_error{"nested task failed"}; }),
                 std::runtime_error);
    EXPECT_EQ(Exec::MaxNumWorkers(), 1);
  });
  EXPECT_EQ(counts, (std::vector<int>{3, 3}));
  EXPECT_EQ(Exec::MaxNumWorkers(), n_threads);
}

TEST(ThreadPool, Grow) {
  ThreadPool pool{"test", 1, [] {}};
  std::vector<std::future<void>> callers;
  for (std::int32_t i = 1; i <= 4; ++i) {
    callers.emplace_back(std::async(std::launch::async, [&, i] { pool.EnsureCapacity(i); }));
  }
  for (auto& caller : callers) {
    caller.get();
  }
  pool.EnsureCapacity(1);
  EXPECT_EQ(pool.NumWorkers(), 4);
}

TEST(ThreadPool, TaskException) {
  ThreadPool pool{"test", 1, [] {}};
  auto void_result = pool.Submit([] { throw std::runtime_error{"task failed"}; });
  EXPECT_THROW(void_result.get(), std::runtime_error);
  auto value_result = pool.Submit([]() -> int { throw std::runtime_error{"task failed"}; });
  EXPECT_THROW(value_result.get(), std::runtime_error);
  // The same worker can still execute tasks after an exception.
  EXPECT_EQ(pool.Submit([] { return 42; }).get(), 42);
}

TEST(ThreadPool, Basic) {
  auto config = *GlobalConfigThreadLocalStore::Get();
  auto original = config.verbosity;
  config.verbosity = 4;
  ThreadPool pool{"test", 2, [config] { *GlobalConfigThreadLocalStore::Get() = config; }};
  EXPECT_EQ(pool.Submit([] { return GlobalConfigThreadLocalStore::Get()->verbosity; }).get(), 4);
  EXPECT_EQ(GlobalConfigThreadLocalStore::Get()->verbosity, original);

  std::vector<std::future<std::size_t>> futures;
  for (std::size_t i = 0; i < 32; ++i) {
    futures.emplace_back(pool.Submit([i] { return i; }));
  }
  for (std::size_t i = 0; i < futures.size(); ++i) {
    EXPECT_EQ(futures[i].get(), i);
  }

  int value{0};
  pool.Submit([&] { value = 3; }).get();
  EXPECT_EQ(value, 3);
}
}  // namespace xgboost::common
