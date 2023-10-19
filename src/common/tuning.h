/**
 * Copyright 2023, XGBoost Contributors
 *
 * @brief Some performance-related compile time constants.
 */
#pragma once

#include <algorithm>  // for min
#include <cstddef>    // for size_t

namespace xgboost::common {
// An ad-hoc estimation of CPU L2 cache size.
constexpr inline double kHistAdHocL2Size = 1024 * 1024 * 0.8;

// Control gradient hardware prefetching during histogram build.
template <typename RowPtrT>
struct HistPrefetch {
 public:
  static constexpr size_t kCacheLineSize = 64;
  static constexpr size_t kPrefetchOffset = 10;

 private:
  constexpr static std::size_t KNoPrefetchSize() {
    return kPrefetchOffset + kCacheLineSize / sizeof(RowPtrT);
  }

 public:
  static size_t NoPrefetchSize(std::size_t rows) { return std::min(rows, KNoPrefetchSize()); }

  template <typename T>
  static constexpr size_t GetPrefetchStep() {
    return HistPrefetch::kCacheLineSize / sizeof(T);
  }
};

// block size of partitioning samples after tree split.
inline constexpr std::size_t kPartitionBlockSize = 2048;

// block size of prediction.
inline constexpr std::size_t kPredictionBlockSize = 64;

// block size of histogram synchronization including subtraction and aggregation from
// threads.
inline constexpr std::size_t kSyncHistBlockSize = 1024;

// block size for build hist.
inline constexpr std::size_t kHistBlockSize = 256;
}  // namespace xgboost::common
