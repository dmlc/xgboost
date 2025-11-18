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
  constexpr static int64_t kUninitCache = -1;
  constexpr static int kMaxCacheSize = 4;
  std::array<int64_t, kMaxCacheSize> cache_size_ = {kUninitCache, kUninitCache,
                                                    kUninitCache, kUninitCache};

  constexpr static int64_t kDefaultL1Size = 32 * 1024; // 32KB
  constexpr static int64_t kDefaultL2Size = 1024 * 1024; // 1MB

  // If CPUID cannot be used, fall back to default L1/L2 cache sizes.
  void SetDefaultCaches() {
    // Overestimating cache sizes harms performance more than underestimation,
    // so conservative defaults are used.
    cache_size_[0] = kDefaultL1Size;
    cache_size_[1] = kDefaultL2Size;
  }

 public:
  CacheManager();

  int64_t L1Size() const {
    return cache_size_[0] != kUninitCache ? cache_size_[0] : kDefaultL1Size;
  }

  int64_t L2Size() const {
    return cache_size_[1] != kUninitCache ? cache_size_[1] : kDefaultL2Size;
  }
};

}  // namespace xgboost::common
#endif  // XGBOOST_COMMON_CACHE_MANAGER_H_

