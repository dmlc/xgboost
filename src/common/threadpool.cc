/**
 * Copyright 2026, XGBoost Contributors
 */
#include "threadpool.h"

#include "threading_utils.h"        // for NameThread
#include "xgboost/global_config.h"  // for InitNewThread

namespace xgboost::common {
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
