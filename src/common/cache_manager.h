/**
 * Copyright 2021-2025, XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_CACHE_MANAGER_H_
#define XGBOOST_COMMON_CACHE_MANAGER_H_

#include <cstdint>     // for int64_t
#include <array>

namespace xgboost::common {

class CacheManager {
 private:
  constexpr static int kMaxCacheSize = 4;
  std::array<int64_t, kMaxCacheSize> cache_size_;

  void SetDefaultCaches() {
    cache_size_[0] = 32 * 1024;    // L1
    cache_size_[1] = 1024 * 1024;  // L2
    cache_size_[2] = 0;            // L3 place holder
    cache_size_[3] = 0;            // L4 place holder
  }

 public:
  CacheManager();

  int64_t L1Size() const {
    return cache_size_[0];
  }

  int64_t L2Size() const {
    return cache_size_[1];
  }
};

}  // namespace xgboost::common
#endif  // XGBOOST_COMMON_CACHE_MANAGER_H_

