/*!
 * Copyright 2015-2019 by Contributors
 * \file common.h
 * \brief Threading utilities
 */
#ifndef XGBOOST_COMMON_THREADING_UTILS_H_
#define XGBOOST_COMMON_THREADING_UTILS_H_

#include <vector>
#include <algorithm>

#include "xgboost/logging.h"

namespace xgboost {
namespace common {

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
// it helps  to improve CPU resources utilization
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
  template<typename Func>
  BlockedSpace2d(size_t dim1, Func getter_size_dim2, size_t grain_size) {
    for (size_t i = 0; i < dim1; ++i) {
      const size_t size = getter_size_dim2(i);
      const size_t n_blocks = size/grain_size + !!(size % grain_size);
      for (size_t iblock = 0; iblock < n_blocks; ++iblock) {
        const size_t begin = iblock * grain_size;
        const size_t end   = std::min(begin + grain_size, size);
        AddBlock(i, begin, end);
      }
    }
  }

  // Amount of blocks(tasks) in a space
  size_t Size() const {
    return ranges_.size();
  }

  // get index of the first dimension of i-th block(task)
  size_t GetFirstDimension(size_t i) const {
    CHECK_LT(i, first_dimension_.size());
    return first_dimension_[i];
  }

  // get a range of indexes for the second dimension of i-th block(task)
  Range1d GetRange(size_t i) const {
    CHECK_LT(i, ranges_.size());
    return ranges_[i];
  }

 private:
  void AddBlock(size_t first_dimension, size_t begin, size_t end) {
    first_dimension_.push_back(first_dimension);
    ranges_.emplace_back(begin, end);
  }

  std::vector<Range1d> ranges_;
  std::vector<size_t> first_dimension_;
};


// Wrapper to implement nested parallelism with simple omp parallel for
template<typename Func>
void ParallelFor2d(const BlockedSpace2d& space, int nthreads, Func func) {
  const size_t num_blocks_in_space = space.Size();
  nthreads = std::min(nthreads, omp_get_max_threads());
  nthreads = std::max(nthreads, 1);

#pragma omp parallel num_threads(nthreads)
  {
    size_t tid = omp_get_thread_num();
    size_t chunck_size = num_blocks_in_space / nthreads + !!(num_blocks_in_space % nthreads);

    size_t begin = chunck_size * tid;
    size_t end   = std::min(begin + chunck_size, num_blocks_in_space);
    for (auto i = begin; i < end; i++) {
      func(space.GetFirstDimension(i), space.GetRange(i));
    }
  }
}

}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_THREADING_UTILS_H_
