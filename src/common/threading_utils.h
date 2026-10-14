/**
 * Copyright 2019-2026, XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_THREADING_UTILS_H_
#define XGBOOST_COMMON_THREADING_UTILS_H_

#include <algorithm>    // for fill_n, min
#include <cstddef>      // for size_t
#include <cstdint>      // for int32_t
#include <thread>       // for thread
#include <type_traits>  // for is_integral_v, is_void_v, invoke_result_t
#include <utility>      // for forward, move
#include <vector>       // for vector

#include "common.h"               // for DivRoundUp
#include "executor.h"             // for Sched
#include "xgboost/logging.h"      // for CHECK_LT
#include "xgboost/string_view.h"  // for StringView

#if defined(XGBOOST_USE_THREADPOOL)
#include "threadpool.h"  // for GlobalThreadPoolExecutor
#else
#include "openmp_exec.h"  // for OpenMPExecutor
#endif

namespace xgboost::common {
#if defined(XGBOOST_USE_THREADPOOL)
using Executor = GlobalThreadPoolExecutor;
#else
using Executor = OpenMPExecutor;
#endif

// Represent simple range of indexes [begin, end)
// Inspired by tbb::blocked_range
class Range1d {
 public:
  Range1d(std::size_t begin, std::size_t end) : begin_{begin}, end_{end} {
    // Avoid CHECK_LT's error-message temporary on the hot path.
    CHECK(begin < end);
  }

  [[nodiscard]] std::size_t begin() const {  // NOLINT
    return begin_;
  }
  [[nodiscard]] std::size_t end() const {  // NOLINT
    return end_;
  }
  [[nodiscard]] std::size_t Size() const { return this->end() - this->begin(); }

 private:
  std::size_t begin_;
  std::size_t end_;
};

// Split 2d space to balanced blocks
// Implementation of the class is inspired by tbb::blocked_range2d
// However, TBB provides only (n x m) 2d range (matrix) separated by blocks. Example:
// [ 1,2,3 ]
// [ 4,5,6 ]
// [ 7,8,9 ]
// But the class is able to work with different sizes in each 'row'. Example:
// [ 1,2 ]
// [ 3,4,5,6 ]
// [ 7,8,9]
// If grain_size is 2: It produces following blocks:
// [1,2], [3,4], [5,6], [7,8], [9]
// The class helps to process data in several tree nodes (non-balanced usually) in parallel
// Using nested parallelism (by nodes and by data in each node)
// it helps to improve CPU resources utilization
class BlockedSpace2d {
 public:
  // Example of space:
  // [ 1,2 ]
  // [ 3,4,5,6 ]
  // [ 7,8,9]
  // BlockedSpace2d will create following blocks (tasks) if grain_size=2:
  // 1-block: first_dimension = 0, range of indexes in a 'row' = [0,2) (includes [1,2] values)
  // 2-block: first_dimension = 1, range of indexes in a 'row' = [0,2) (includes [3,4] values)
  // 3-block: first_dimension = 1, range of indexes in a 'row' = [2,4) (includes [5,6] values)
  // 4-block: first_dimension = 2, range of indexes in a 'row' = [0,2) (includes [7,8] values)
  // 5-block: first_dimension = 2, range of indexes in a 'row' = [2,3) (includes [9] values)
  // Arguments:
  // dim1 - size of the first dimension in the space
  // getter_size_dim2 - functor to get the second dimensions for each 'row' by row-index
  // grain_size - max size of produced blocks
  template <typename Getter>
  BlockedSpace2d(std::size_t dim1, Getter&& getter_size_dim2, std::size_t grain_size) {
    static_assert(std::is_integral_v<std::invoke_result_t<Getter, std::size_t>>);
    for (std::size_t i = 0; i < dim1; ++i) {
      std::size_t size = getter_size_dim2(i);
      // Each row (second dim) is divided into n_blocks
      std::size_t n_blocks = size / grain_size + !!(size % grain_size);
      for (std::size_t iblock = 0; iblock < n_blocks; ++iblock) {
        std::size_t begin = iblock * grain_size;
        std::size_t end = std::min(begin + grain_size, size);
        AddBlock(i, begin, end);
      }
    }
  }

  // Amount of blocks(tasks) in a space
  [[nodiscard]] std::size_t Size() const { return ranges_.size(); }

  // get index of the first dimension of i-th block(task)
  [[nodiscard]] std::size_t GetFirstDimension(std::size_t i) const {
    CHECK_LT(i, first_dimension_.size());
    return first_dimension_[i];
  }

  // get a range of indexes for the second dimension of i-th block(task)
  [[nodiscard]] Range1d GetRange(std::size_t i) const {
    CHECK_LT(i, ranges_.size());
    return ranges_[i];
  }

 private:
  /**
   * @brief Add a parallel block.
   *
   * @param first_dim The row index.
   * @param begin     The begin of the second dimension.
   * @param end       The end of the second dimension.
   */
  void AddBlock(std::size_t first_dim, std::size_t begin, std::size_t end) {
    first_dimension_.push_back(first_dim);
    ranges_.emplace_back(begin, end);
  }

  std::vector<Range1d> ranges_;
  std::vector<std::size_t> first_dimension_;
};

template <typename Func>
void ParallelFor2d(const BlockedSpace2d& space, std::int32_t n_threads, Func&& fn) {
  static_assert(std::is_void_v<std::invoke_result_t<Func, std::size_t, Range1d>>);
  Executor::Execute2d(space, n_threads, std::forward<Func>(fn));
}

template <typename Fn>
void ParallelFor2d(BlockedSpace2d const& space, std::int32_t n_threads, WithWorker<Fn>&& fn) {
  static_assert(std::is_void_v<std::invoke_result_t<Fn, std::size_t, Range1d, Worker>>);
  Executor::Execute2d(space, n_threads, fn);
}

template <typename Index, typename Func>
void ParallelFor(Index size, std::int32_t n_threads, Sched sched, Func&& fn) {
  if (n_threads == 1) {
    // early exit
    auto&& callback = detail::BindWorker(fn, [] { return Worker{0}; });
    for (Index i = 0; i < size; ++i) {
      callback(i);
    }
    return;
  }

  Executor::Execute(size, n_threads, sched, std::forward<Func>(fn));
}

template <typename Index, typename Func>
void ParallelFor(Index size, std::int32_t n_threads, Func&& fn) {
  ParallelFor(size, n_threads, Sched::Static(), std::forward<Func>(fn));
}

/**
 * @brief 1-d block-based parallel for loop.
 *
 * @tparam kBlockOfRowsSize The size of the block.
 * @tparam Index The type of the index.
 * @tparam Func The type of the function.
 *
 * @param size The size of the range.
 * @param n_threads The number of threads.
 * @param fn The function to execute. The function should take a Range1d as an argument.
 */
template <std::size_t kBlockOfRowsSize, typename Index, typename Func>
void ParallelFor1d(Index size, std::int32_t n_threads, Func&& fn) {
  auto const n_blocks = DivRoundUp(size, kBlockOfRowsSize);
  common::ParallelFor(
      n_blocks, n_threads, detail::TransformCallback(fn, [&](auto& callback, auto block_id) {
        static_assert(std::is_void_v<std::invoke_result_t<decltype(callback), Range1d>>);
        std::size_t const block_beg = block_id * kBlockOfRowsSize;
        auto const block_size =
            std::min(static_cast<std::size_t>(size - block_beg), kBlockOfRowsSize);
        callback(Range1d{block_beg, block_beg + block_size});
      }));
}

/** @brief Use n_threads as the number of blocks. */
template <typename Index, typename Func>
void ParallelForBlock(Index size, std::int32_t n_threads, Func&& fn) {
  std::size_t blk_size = size / n_threads + (size % n_threads > 0);
  ParallelFor(n_threads, n_threads, detail::TransformCallback(fn, [&](auto& callback, auto tid) {
                static_assert(std::is_void_v<std::invoke_result_t<decltype(callback), Range1d>>);
                auto blk_beg = tid * blk_size;
                auto blk_end = std::min((tid + 1) * blk_size, static_cast<std::size_t>(size));
                if (blk_end <= blk_beg) {
                  return;
                }
                callback(Range1d{blk_beg, blk_end});
              }));
}

/**
 * @brief Get thread limit from CFS.
 *
 * This function has non-trivial overhead and should not be called repeatedly.
 */
std::int32_t GetCfsCPUCount() noexcept;

/**
 * @brief Get the number of available threads based on n_threads specified by users.
 */
std::int32_t GetNumThreads(std::int32_t n_threads) noexcept(true);

/**
 * @brief A C-style array with in-stack allocation.
 *
 * As long as the array is smaller than MaxStackSize, it will be allocated inside the
 * stack. Otherwise, it will be heap-allocated.
 */
template <typename T, std::size_t MaxStackSize>
class MemStackAllocator {
 public:
  explicit MemStackAllocator(size_t required_size) : required_size_(required_size) {
    if (MaxStackSize >= required_size_) {
      ptr_ = stack_mem_;
    } else {
      ptr_ = new T[required_size_];
    }
  }
  MemStackAllocator(size_t required_size, T init) : MemStackAllocator{required_size} {
    std::fill_n(ptr_, required_size_, init);
  }

  ~MemStackAllocator() {
    if (required_size_ > MaxStackSize) {
      delete[] ptr_;
    }
  }
  T& operator[](size_t i) { return ptr_[i]; }
  T const& operator[](size_t i) const { return ptr_[i]; }

  auto data() const { return ptr_; }                   // NOLINT
  auto data() { return ptr_; }                         // NOLINT
  std::size_t size() const { return required_size_; }  // NOLINT

  auto cbegin() const { return data(); }         // NOLINT
  auto cend() const { return data() + size(); }  // NOLINT

 private:
  T* ptr_ = nullptr;
  size_t required_size_;
  T stack_mem_[MaxStackSize];
};

/**
 * @brief Constant that can be used for initializing static thread local memory.
 */
std::int32_t constexpr DefaultMaxThreads() { return 128; }

/**
 * @brief Give the thread a name. Supports only pthread on linux.
 */
void NameThread(std::thread* t, StringView name);
}  // namespace xgboost::common

#endif  // XGBOOST_COMMON_THREADING_UTILS_H_
