/**
 * Copyright 2019-2023 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_THREADING_UTILS_H_
#define XGBOOST_COMMON_THREADING_UTILS_H_

#include <dmlc/common.h>
#include <dmlc/omp.h>

#include <algorithm>    // for min
#include <cstddef>      // for size_t
#include <cstdint>      // for int32_t
#include <cstdlib>      // for malloc, free
#include <functional>   // for function
#include <new>          // for bad_alloc
#include <type_traits>  // for is_signed, conditional_t, is_integral_v, invoke_result_t
#include <vector>       // for vector

#include "xgboost/logging.h"

#if !defined(_OPENMP)
extern "C" {
inline int32_t omp_get_thread_limit() __GOMP_NOTHROW { return 1; }  // NOLINT
}
#endif  // !defined(_OPENMP)

// MSVC doesn't implement the thread limit.
#if defined(_OPENMP) && defined(_MSC_VER)
#include <limits>

extern "C" {
inline int32_t omp_get_thread_limit() { return std::numeric_limits<int32_t>::max(); }  // NOLINT
}
#endif  // defined(_MSC_VER)

namespace xgboost::common {
// Represent simple range of indexes [begin, end)
// Inspired by tbb::blocked_range
class Range1d {
 public:
  Range1d(size_t begin, size_t end): begin_(begin), end_(end) {
    CHECK_LT(begin, end);
  }

  size_t begin() const {  // NOLINT
    return begin_;
  }

  size_t end() const {  // NOLINT
    return end_;
  }

 private:
  size_t begin_;
  size_t end_;
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
  [[nodiscard]] std::size_t Size() const {
    return ranges_.size();
  }

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


// Wrapper to implement nested parallelism with simple omp parallel for
template <typename Func>
void ParallelFor2d(const BlockedSpace2d& space, int n_threads, Func&& func) {
  static_assert(std::is_void_v<std::invoke_result_t<Func, std::size_t, Range1d>>);
  std::size_t n_blocks_in_space = space.Size();
  CHECK_GE(n_threads, 1);

  dmlc::OMPException exc;
#pragma omp parallel num_threads(n_threads)
  {
    exc.Run([&]() {
      std::size_t tid = omp_get_thread_num();
      std::size_t chunck_size = n_blocks_in_space / n_threads + !!(n_blocks_in_space % n_threads);

      std::size_t begin = chunck_size * tid;
      std::size_t end = std::min(begin + chunck_size, n_blocks_in_space);
      for (auto i = begin; i < end; i++) {
        func(space.GetFirstDimension(i), space.GetRange(i));
      }
    });
  }
  exc.Rethrow();
}

/**
 * OpenMP schedule
 */
struct Sched {
  enum {
    kAuto,
    kDynamic,
    kStatic,
    kGuided,
  } sched;
  size_t chunk{0};

  Sched static Auto() { return Sched{kAuto}; }
  Sched static Dyn(size_t n = 0) { return Sched{kDynamic, n}; }
  Sched static Static(size_t n = 0) { return Sched{kStatic, n}; }
  Sched static Guided() { return Sched{kGuided}; }
};

template <typename Index, typename Func>
void ParallelFor(Index size, int32_t n_threads, Sched sched, Func fn) {
#if defined(_MSC_VER)
  // msvc doesn't support unsigned integer as openmp index.
  using OmpInd = std::conditional_t<std::is_signed<Index>::value, Index, omp_ulong>;
#else
  using OmpInd = Index;
#endif
  OmpInd length = static_cast<OmpInd>(size);
  CHECK_GE(n_threads, 1);

  dmlc::OMPException exc;
  switch (sched.sched) {
  case Sched::kAuto: {
#pragma omp parallel for num_threads(n_threads)
    for (OmpInd i = 0; i < length; ++i) {
      exc.Run(fn, i);
    }
    break;
  }
  case Sched::kDynamic: {
    if (sched.chunk == 0) {
#pragma omp parallel for num_threads(n_threads) schedule(dynamic)
      for (OmpInd i = 0; i < length; ++i) {
        exc.Run(fn, i);
      }
    } else {
#pragma omp parallel for num_threads(n_threads) schedule(dynamic, sched.chunk)
      for (OmpInd i = 0; i < length; ++i) {
        exc.Run(fn, i);
      }
    }
    break;
  }
  case Sched::kStatic: {
    if (sched.chunk == 0) {
#pragma omp parallel for num_threads(n_threads) schedule(static)
      for (OmpInd i = 0; i < length; ++i) {
        exc.Run(fn, i);
      }
    } else {
#pragma omp parallel for num_threads(n_threads) schedule(static, sched.chunk)
      for (OmpInd i = 0; i < length; ++i) {
        exc.Run(fn, i);
      }
    }
    break;
  }
  case Sched::kGuided: {
#pragma omp parallel for num_threads(n_threads) schedule(guided)
    for (OmpInd i = 0; i < length; ++i) {
      exc.Run(fn, i);
    }
    break;
  }
  }
  exc.Rethrow();
}

template <typename Index, typename Func>
void ParallelFor(Index size, int32_t n_threads, Func fn) {
  ParallelFor(size, n_threads, Sched::Static(), fn);
}

inline std::int32_t OmpGetThreadLimit() {
  std::int32_t limit = omp_get_thread_limit();
  CHECK_GE(limit, 1) << "Invalid thread limit for OpenMP.";
  return limit;
}

/**
 * \brief Get thread limit from CFS.
 *
 *   This function has non-trivial overhead and should not be called repeatly.
 */
std::int32_t GetCfsCPUCount() noexcept;

/**
 * @brief Get the number of available threads based on n_threads specified by users.
 */
std::int32_t OmpGetNumThreads(std::int32_t n_threads) noexcept(true);

/*!
 * \brief A C-style array with in-stack allocation. As long as the array is smaller than
 * MaxStackSize, it will be allocated inside the stack. Otherwise, it will be
 * heap-allocated.
 */
template <typename T, std::size_t MaxStackSize>
class MemStackAllocator {
 public:
  explicit MemStackAllocator(size_t required_size) : required_size_(required_size) {
    if (MaxStackSize >= required_size_) {
      ptr_ = stack_mem_;
    } else {
      ptr_ = reinterpret_cast<T*>(std::malloc(required_size_ * sizeof(T)));
    }
    if (!ptr_) {
      throw std::bad_alloc{};
    }
  }
  MemStackAllocator(size_t required_size, T init) : MemStackAllocator{required_size} {
    std::fill_n(ptr_, required_size_, init);
  }

  ~MemStackAllocator() {
    if (required_size_ > MaxStackSize) {
      std::free(ptr_);
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
 * \brief Constant that can be used for initializing static thread local memory.
 */
std::int32_t constexpr DefaultMaxThreads() { return 128; }
}  // namespace xgboost::common

#endif  // XGBOOST_COMMON_THREADING_UTILS_H_
