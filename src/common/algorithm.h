/**
 * Copyright 2022-2023 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_ALGORITHM_H_
#define XGBOOST_COMMON_ALGORITHM_H_
#include <algorithm>          // std::upper_bound,std::stable_sort,std::sort,max
#include <cinttypes>          // std::size_t
#include <iterator>           // std::iterator_traits,std::distance
#include <vector>             // std::vector

#include "numeric.h"          // Iota
#include "xgboost/base.h"     // GCC_HAS_PARALLEL,MSVC_HAS_PARALLEL
#include "xgboost/context.h"  // Context

#if GCC_HAS_PARALLEL()
#include <parallel/algorithm>
#elif defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#include <ppl.h>
#else
// nothing
#endif  // GLIBC VERSION

namespace xgboost {
namespace common {
template <typename It, typename Idx>
auto SegmentId(It first, It last, Idx idx) {
  std::size_t segment_id = std::upper_bound(first, last, idx) - 1 - first;
  return segment_id;
}

template <typename Iter, typename Comp>
void StableSort(Context const *ctx, Iter begin, Iter end, Comp &&comp) {
  if (ctx->Threads() > 1) {
#if GCC_HAS_PARALLEL()
    __gnu_parallel::stable_sort(begin, end, comp,
                                __gnu_parallel::default_parallel_tag(ctx->Threads()));
#else
    std::stable_sort(begin, end, comp);
#endif  // GLIBC VERSION
  } else {
    std::stable_sort(begin, end, comp);
  }
}

template <typename Iter, typename Comp>
void Sort(Context const *ctx, Iter begin, Iter end, Comp comp) {
  if (ctx->Threads() > 1) {
#if GCC_HAS_PARALLEL()
    __gnu_parallel::sort(begin, end, comp, __gnu_parallel::default_parallel_tag(ctx->Threads()));
#elif defined(_MSC_VER) && !defined(__INTEL_COMPILER)
    auto n = std::distance(begin, end);
    // use chunk size as hint to number of threads. No local policy/scheduler input with the
    // concurrency module.
    std::size_t chunk_size = n / ctx->Threads();
    chunk_size = std::max(chunk_size, static_cast<std::size_t>(2048));
    concurrency::parallel_sort(begin, end, comp, chunk_size);
#else
    std::cout << "msvc no concurrency" << std::endl;
    std::sort(begin, end, comp);
#endif  // GLIBC VERSION
  } else {
    std::cout << "1 thread" << std::endl;
    std::sort(begin, end, comp);
  }
}

template <typename Idx, typename Iter, typename V = typename std::iterator_traits<Iter>::value_type,
          typename Comp = std::less<V>>
std::vector<Idx> ArgSort(Context const *ctx, Iter begin, Iter end, Comp comp = std::less<V>{}) {
  CHECK(ctx->IsCPU());
  auto n = std::distance(begin, end);
  std::vector<Idx> result(n);
  Iota(ctx, result.begin(), result.end(), 0);
  auto op = [&](Idx const &l, Idx const &r) { return comp(begin[l], begin[r]); };
  Sort(ctx, result.begin(), result.end(), op);
  return result;
}
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_ALGORITHM_H_
