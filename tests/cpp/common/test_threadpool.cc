/**
 * Copyright 2024, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/global_config.h>  // for GlobalConfigThreadLocalStore

#include <cstddef>  // for size_t
#include <cstdint>  // for int32_t
#include <future>   // for future
#include <thread>   // for sleep_for, thread

#include "../../../src/common/threadpool.h"

namespace xgboost::common {
TEST(ThreadPool, Basic) {
  std::int32_t n_threads = std::thread::hardware_concurrency();

  // Set verbosity to 0 for thread-local variable.
  auto orig = GlobalConfigThreadLocalStore::Get()->verbosity;
  GlobalConfigThreadLocalStore::Get()->verbosity = 4;
  // 4 is an invalid value, it's only possible to set it by bypassing the parameter
  // validation.
  ASSERT_NE(orig, GlobalConfigThreadLocalStore::Get()->verbosity);
  ThreadPool pool{n_threads, [config = *GlobalConfigThreadLocalStore::Get()] {
                    *GlobalConfigThreadLocalStore::Get() = config;
                  }};
  GlobalConfigThreadLocalStore::Get()->verbosity = orig;  // restore

  {
    auto fut = pool.Submit([] { return GlobalConfigThreadLocalStore::Get()->verbosity; });
    ASSERT_EQ(fut.get(), 4);
    ASSERT_EQ(GlobalConfigThreadLocalStore::Get()->verbosity, orig);
  }
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
  {
    std::int32_t val{0};
    auto fut = pool.Submit([&] { val = 3; });
    static_assert(std::is_void_v<decltype(fut.get())>);
    fut.get();
    ASSERT_EQ(val, 3);
  }
}
}  // namespace xgboost::common
