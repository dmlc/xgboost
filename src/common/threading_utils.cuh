/**
 * Copyright 2021-2023 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_THREADING_UTILS_CUH_
#define XGBOOST_COMMON_THREADING_UTILS_CUH_

#include <algorithm>           // std::min
#include <cstddef>             // std::size_t

#include "./math.h"            // Sqr
#include "common.h"
#include "device_helpers.cuh"  // LaunchN
#include "xgboost/base.h"      // XGBOOST_DEVICE
#include "xgboost/span.h"      // Span

namespace xgboost {
namespace common {
/**
 * \param n Number of items (length of the base)
 * \param h hight
 */
XGBOOST_DEVICE inline std::size_t DiscreteTrapezoidArea(std::size_t n, std::size_t h) {
  n -= 1;              // without diagonal entries
  h = std::min(n, h);  // Used for ranking, h <= n
  std::size_t total = ((n - (h - 1)) + n) * h / 2;
  return total;
}

/**
 * Used for mapping many groups of trapezoid shaped computation onto CUDA blocks.  The
 * trapezoid must be on upper right corner.
 *
 * Equivalent to loops like:
 *
 * \code
 *   for (std::size_t i = 0; i < h; ++i) {
 *     for (std::size_t j = i + 1; j < n; ++j) {
 *        do_something();
 *     }
 *   }
 * \endcode
 *
 * with h <= n
 */
template <typename U>
std::size_t SegmentedTrapezoidThreads(xgboost::common::Span<U> group_ptr,
                                      xgboost::common::Span<std::size_t> out_group_threads_ptr,
                                      std::size_t h) {
  CHECK_GE(group_ptr.size(), 1);
  CHECK_EQ(group_ptr.size(), out_group_threads_ptr.size());
  dh::LaunchN(group_ptr.size(), [=] XGBOOST_DEVICE(std::size_t idx) {
    if (idx == 0) {
      out_group_threads_ptr[0] = 0;
      return;
    }

    std::size_t cnt = static_cast<std::size_t>(group_ptr[idx] - group_ptr[idx - 1]);
    out_group_threads_ptr[idx] = DiscreteTrapezoidArea(cnt, h);
  });
  dh::InclusiveSum(out_group_threads_ptr.data(), out_group_threads_ptr.data(),
                   out_group_threads_ptr.size());
  std::size_t total = 0;
  dh::safe_cuda(cudaMemcpy(&total, out_group_threads_ptr.data() + out_group_threads_ptr.size() - 1,
                           sizeof(total), cudaMemcpyDeviceToHost));
  return total;
}

/**
 * Called inside kernel to obtain coordinate from trapezoid grid.
 */
XGBOOST_DEVICE inline void UnravelTrapeziodIdx(std::size_t i_idx, std::size_t n, std::size_t *out_i,
                                               std::size_t *out_j) {
  auto &i = *out_i;
  auto &j = *out_j;
  double idx = static_cast<double>(i_idx);
  double N = static_cast<double>(n);

  i = std::ceil(-(0.5 - N + std::sqrt(common::Sqr(N - 0.5) + 2.0 * (-idx - 1.0)))) - 1.0;

  auto I = static_cast<double>(i);
  size_t n_elems = -0.5 * common::Sqr(I) + (N - 0.5) * I;

  j = idx - n_elems + i + 1;
}
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_THREADING_UTILS_CUH_
