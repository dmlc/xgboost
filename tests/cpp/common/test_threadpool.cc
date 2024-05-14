/**
 * Copyright 2024, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <cstddef>  // for size_t
#include <cstdint>  // for int32_t
#include <future>   // for future
#include <thread>   // for sleep_for, thread

#include "../../../src/common/threadpool.h"

namespace xgboost::common {
TEST(ThreadPool, Basic) {
  std::int32_t n_threads = std::thread::hardware_concurrency();
  ThreadPool pool{n_threads};
  {
    auto fut = pool.Submit([] { return 3; });
    ASSERT_EQ(fut.get(), 3);
  }
  {
    auto fut = pool.Submit([] { return std::string{"ok"}; });
    ASSERT_EQ(fut.get(), "ok");
  }
  {
    std::vector<std::future<std::size_t>> futures;
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_threads) * 16; ++i) {
      futures.emplace_back(pool.Submit([=] {
        std::this_thread::sleep_for(std::chrono::milliseconds{10});
        return i;
      }));
    }
    for (std::size_t i = 0; i < futures.size(); ++i) {
      ASSERT_EQ(futures[i].get(), i);
    }
  }
  {
    std::vector<std::future<std::size_t>> futures;
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_threads) * 16; ++i) {
      futures.emplace_back(pool.Submit([=] {
        return i;
      }));
    }
    for (std::size_t i = 0; i < futures.size(); ++i) {
      ASSERT_EQ(futures[i].get(), i);
    }
  }
}
}  // namespace xgboost::common
