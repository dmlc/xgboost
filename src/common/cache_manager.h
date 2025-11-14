/**
 * Copyright 2021-2025, XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_CACHE_MANAGER_H_
#define XGBOOST_COMMON_CACHE_MANAGER_H_

#include <cstdint>     // for int64_t
#include <array>

namespace xgboost::common {

/* Detect cache sizes at runtime,
 * or fall back to defaults if detection is not possible.
 */
class CacheManager {
 private:
  constexpr static int kMaxCacheSize = 4;
  std::array<int64_t, kMaxCacheSize> cache_size_;

  // If CPUID cannot be used, fall back to default L1/L2 cache sizes.
  void SetDefaultCaches() {
    // Overestimating cache sizes harms performance more than underestimation,
    // so conservative defaults are used.
    cache_size_[0] = 32 * 1024;    // L1, 32KB
    cache_size_[1] = 1024 * 1024;  // L2, 1MB
    cache_size_[2] = 0;            // L3, place holder, not used
    cache_size_[3] = 0;            // L4, place holder, not used
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

