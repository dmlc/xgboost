/**
 * Copyright 2026, XGBoost Contributors
 */
#include "threadpool.h"

#include <algorithm>  // for max, min
#include <cstdint>    // for int32_t
#include <mutex>      // for unique_lock
#include <string>     // for string, to_string
#include <thread>     // for thread

#include "threading_utils.h"        // for NameThread
#include "xgboost/global_config.h"  // for GlobalConfigThreadLocalStore, InitNewThread

namespace xgboost::common {
std::int32_t GlobalThreadPoolExecutor::MaxNumWorkers() noexcept {
  if (in_parallel_) {
    return 1;
  }
  static auto max_n_threads = std::thread::hardware_concurrency();
  auto limit = std::max(1u, max_n_threads);
  auto configured = GlobalConfigThreadLocalStore::Get()->nthread;
  return configured > 0 ? std::min(limit, static_cast<unsigned>(configured)) : limit;
}

void ThreadPool::EnsureCapacity(std::int32_t n_threads) {
  std::unique_lock lock{grow_mu_};
  for (auto i = static_cast<std::int32_t>(pool_.size()); i < n_threads; ++i) {
    pool_.emplace_back(worker_fn_);
    std::string name_i = name_ + "-" + std::to_string(i);
    NameThread(&pool_.back(), name_i);
  }
}

ThreadPool* GetGlobalThreadPool(std::int32_t n_threads) {
  static ThreadPool pool("te", 1, InitNewThread{});
  pool.EnsureCapacity(n_threads);
  return &pool;
}
}  // namespace xgboost::common
