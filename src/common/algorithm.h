/**
 * Copyright 2022-2023 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_ALGORITHM_H_
#define XGBOOST_COMMON_ALGORITHM_H_
#include <algorithm>          // upper_bound, stable_sort, sort, max
#include <cinttypes>          // size_t
#include <functional>         // less
#include <iterator>           // iterator_traits, distance
#include <vector>             // vector

#include "numeric.h"          // Iota
#include "xgboost/context.h"  // Context

// clang with libstdc++ works as well
#if defined(__GNUC__) && (__GNUC__ >= 4) && !defined(__sun) && !defined(sun) && \
    !defined(__APPLE__) && __has_include(<omp.h>) && __has_include(<parallel/algorithm>)
#define GCC_HAS_PARALLEL 1
#endif  // GLIC_VERSION

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#define MSVC_HAS_PARALLEL 1
#endif  // MSC

#if defined(GCC_HAS_PARALLEL)
#include <parallel/algorithm>
#elif defined(MSVC_HAS_PARALLEL)
#include <ppl.h>
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
#if defined(GCC_HAS_PARALLEL)
    __gnu_parallel::stable_sort(begin, end, comp,
                                __gnu_parallel::default_parallel_tag(ctx->Threads()));
#else
    // the only stable sort is radix sort for msvc ppl.
    std::stable_sort(begin, end, comp);
#endif  // GLIBC VERSION
  } else {
    std::stable_sort(begin, end, comp);
  }
}

template <typename Iter, typename Comp>
void Sort(Context const *ctx, Iter begin, Iter end, Comp comp) {
  if (ctx->Threads() > 1) {
#if defined(GCC_HAS_PARALLEL)
    __gnu_parallel::sort(begin, end, comp, __gnu_parallel::default_parallel_tag(ctx->Threads()));
#elif defined(MSVC_HAS_PARALLEL)
    auto n = std::distance(begin, end);
    // use chunk size as hint to number of threads. No local policy/scheduler input with the
    // concurrency module.
    std::size_t chunk_size = n / ctx->Threads();
    // 2048 is the default of msvc ppl as of v2022.
    chunk_size = std::max(chunk_size, static_cast<std::size_t>(2048));
    concurrency::parallel_sort(begin, end, comp, chunk_size);
#else
    std::sort(begin, end, comp);
#endif  // GLIBC VERSION
  } else {
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
  StableSort(ctx, result.begin(), result.end(), op);
  return result;
}
}  // namespace common
}  // namespace xgboost

#if defined(GCC_HAS_PARALLEL)
#undef GCC_HAS_PARALLEL
#endif  // defined(GCC_HAS_PARALLEL)

#if defined(MSVC_HAS_PARALLEL)
#undef MSVC_HAS_PARALLEL
#endif  // defined(MSVC_HAS_PARALLEL)

#endif  // XGBOOST_COMMON_ALGORITHM_H_
