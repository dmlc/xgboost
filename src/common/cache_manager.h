/**
 * Copyright 2021-2026, XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_CACHE_MANAGER_H_
#define XGBOOST_COMMON_CACHE_MANAGER_H_

#include <array>
#include <cstddef>   // for size_t
#include <cstdint>   // for int64_t
#include <optional>  // for optional

#include "xgboost/span.h"  // for Span

namespace xgboost::common {

namespace detail {
/* Data cache sizes reported for one CPU performance level. An empty field means the
 * platform did not report that quantity.
 */
struct PerfLevelCache {
  std::optional<std::int64_t> l1d;
  std::optional<std::int64_t> l2;
  /* Number of cores sharing one L2, needed because L2 is per-cluster on heterogeneous
   * CPUs rather than private per core as on x86.
   */
  std::optional<std::int64_t> cpus_per_l2;
};

/* The sizes a single thread can rely on, reduced over all performance levels. */
struct DataCacheSizes {
  std::optional<std::int64_t> l1d;
  std::optional<std::int64_t> l2_per_cpu;
};

/* Reduce per-performance-level cache readings to per-thread sizes.
 *
 * Split out from the platform query so the policy can be tested with synthetic readings
 * on any host, including topologies not present on the build machine. See
 * cache_manager.cc for why the minimum is taken and why L2 is divided.
 */
[[nodiscard]] DataCacheSizes ReduceHeterogeneousCaches(Span<PerfLevelCache const> levels);
}  // namespace detail

/* Size of a cache line in bytes.
 *
 * Used to pick prefetch strides and to keep concurrently written data from sharing a
 * line. Apple Silicon uses 128-byte lines; the remaining targets use 64.
 *
 * std::hardware_destructive_interference_size is deliberately not used: libc++ reports 64
 * for it on arm64, including on Apple Silicon, so it would reintroduce the same wrong
 * value this constant exists to correct.
 */
#if defined(__APPLE__) && defined(__aarch64__)
constexpr std::size_t kCacheLineSize = 128;
#else
constexpr std::size_t kCacheLineSize = 64;
#endif  // defined(__APPLE__) && defined(__aarch64__)

/* Detect cache sizes at runtime,
 * or fall back to defaults if detection is not possible.
 */
class CacheManager {
 private:
  constexpr static int64_t kUninitCache = -1;
  constexpr static int kMaxCacheSize = 4;
  std::array<int64_t, kMaxCacheSize> cache_size_ = {kUninitCache, kUninitCache, kUninitCache,
                                                    kUninitCache};

  constexpr static int64_t kDefaultL1Size = 32 * 1024;    // 32KB
  constexpr static int64_t kDefaultL2Size = 1024 * 1024;  // 1MB
  constexpr static int64_t kDefaultL3Size = 0;            // 0MB

  // If no runtime detection is available, fall back to default L1/L2 cache sizes.
  void SetDefaultCaches() {
    // Overestimating cache sizes harms performance more than underestimation,
    // so conservative defaults are used.
    cache_size_[0] = kDefaultL1Size;
    cache_size_[1] = kDefaultL2Size;
    cache_size_[2] = kDefaultL3Size;
  }

 public:
  CacheManager();

  int64_t L1Size() const {
    return cache_size_[0] != kUninitCache ? cache_size_[0] : kDefaultL1Size;
  }

  int64_t L2Size() const {
    return cache_size_[1] != kUninitCache ? cache_size_[1] : kDefaultL2Size;
  }

  int64_t L3Size() const {
    return cache_size_[2] != kUninitCache ? cache_size_[2] : kDefaultL3Size;
  }
};
}  // namespace xgboost::common
#endif  // XGBOOST_COMMON_CACHE_MANAGER_H_
