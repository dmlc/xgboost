/**
 * Copyright 2022-2023 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_ALGORITHM_H_
#define XGBOOST_COMMON_ALGORITHM_H_
#include <algorithm>          // for upper_bound, stable_sort, sort, max, all_of, none_of, min
#include <cstddef>            // for size_t
#include <functional>         // for less
#include <iterator>           // for iterator_traits, distance
#include <type_traits>        // for is_same
#include <vector>             // for vector

#include "common.h"           // for DivRoundUp
#include "numeric.h"          // for Iota
#include "threading_utils.h"  // for MemStackAllocator, DefaultMaxThreads, ParallelFor
#include "xgboost/context.h"  // for Context

// clang with libstdc++ works as well
#if defined(__GNUC__) && (__GNUC__ >= 4) && !defined(__sun) && !defined(sun) && \
    !defined(__APPLE__) && defined(_OPENMP)
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

namespace detail {
template <typename It, typename Op>
bool Logical(Context const *ctx, It first, It last, Op op) {
  auto n = std::distance(first, last);
  auto n_threads =
      std::max(std::min(n, static_cast<decltype(n)>(ctx->Threads())), static_cast<decltype(n)>(1));
  common::MemStackAllocator<bool, common::DefaultMaxThreads()> tloc{
      static_cast<std::size_t>(n_threads), false};
  CHECK_GE(n, 0);
  CHECK_GE(ctx->Threads(), 1);
  static_assert(std::is_same<decltype(n), decltype(n_threads)>::value, "");
  auto const n_per_thread = common::DivRoundUp(n, ctx->Threads());
  common::ParallelFor(static_cast<decltype(n)>(n_threads), n_threads, [&](auto t) {
    auto begin = t * n_per_thread;
    auto end = std::min(begin + n_per_thread, n);

    auto first_tloc = first + begin;
    auto last_tloc = first + end;
    if (first_tloc >= last_tloc) {
      tloc[t] = true;
      return;
    }
    bool result = op(first_tloc, last_tloc);
    tloc[t] = result;
  });
  return std::all_of(tloc.cbegin(), tloc.cend(), [](auto v) { return v; });
}
}  // namespace detail

/**
 * \brief Parallel version of std::none_of
 */
template <typename It, typename Pred>
bool NoneOf(Context const *ctx, It first, It last, Pred predicate) {
  return detail::Logical(ctx, first, last, [&predicate](It first, It last) {
    return std::none_of(first, last, predicate);
  });
}

/**
 * \brief Parallel version of std::all_of
 */
template <typename It, typename Pred>
bool AllOf(Context const *ctx, It first, It last, Pred predicate) {
  return detail::Logical(ctx, first, last, [&predicate](It first, It last) {
    return std::all_of(first, last, predicate);
  });
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
