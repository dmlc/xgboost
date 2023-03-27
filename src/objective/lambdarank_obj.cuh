/**
 * Copyright 2023 XGBoost contributors
 */
#ifndef XGBOOST_OBJECTIVE_LAMBDARANK_OBJ_CUH_
#define XGBOOST_OBJECTIVE_LAMBDARANK_OBJ_CUH_

#include <thrust/binary_search.h>                      // for lower_bound, upper_bound
#include <thrust/functional.h>                         // for greater
#include <thrust/iterator/counting_iterator.h>         // for make_counting_iterator
#include <thrust/random/linear_congruential_engine.h>  // for minstd_rand
#include <thrust/random/uniform_int_distribution.h>    // for uniform_int_distribution

#include <cassert>                                     // for cassert
#include <cstddef>                                     // for size_t
#include <cstdint>                                     // for int32_t
#include <tuple>                                       // for make_tuple, tuple

#include "../common/device_helpers.cuh"                // for MakeTransformIterator
#include "../common/ranking_utils.cuh"                 // for PairsForGroup
#include "../common/ranking_utils.h"                   // for RankingCache
#include "../common/threading_utils.cuh"               // for UnravelTrapeziodIdx
#include "xgboost/base.h"    // for bst_group_t, GradientPair, XGBOOST_DEVICE
#include "xgboost/data.h"    // for MetaInfo
#include "xgboost/linalg.h"  // for VectorView, Range, UnravelIndex
#include "xgboost/span.h"    // for Span

namespace xgboost::obj::cuda_impl {
/**
 * \brief Find number of elements left to the label bucket
 */
template <typename It, typename T = typename std::iterator_traits<It>::value_type>
XGBOOST_DEVICE __forceinline__ std::size_t CountNumItemsToTheLeftOf(It items, std::size_t n, T v) {
  return thrust::lower_bound(thrust::seq, items, items + n, v, thrust::greater<T>{}) - items;
}
/**
 * \brief Find number of elements right to the label bucket
 */
template <typename It, typename T = typename std::iterator_traits<It>::value_type>
XGBOOST_DEVICE __forceinline__ std::size_t CountNumItemsToTheRightOf(It items, std::size_t n, T v) {
  return n - (thrust::upper_bound(thrust::seq, items, items + n, v, thrust::greater<T>{}) - items);
}
/**
 * \brief Sort labels according to rank list for making pairs.
 */
common::Span<std::size_t const> SortY(Context const *ctx, MetaInfo const &info,
                                      common::Span<std::size_t const> d_rank,
                                      std::shared_ptr<ltr::RankingCache> p_cache);

/**
 * \brief Parameters needed for calculating gradient
 */
struct KernelInputs {
  linalg::VectorView<double const> ti_plus;   // input bias ratio
  linalg::VectorView<double const> tj_minus;  // input bias ratio
  linalg::VectorView<double> li;
  linalg::VectorView<double> lj;

  common::Span<bst_group_t const> d_group_ptr;
  common::Span<std::size_t const> d_threads_group_ptr;
  common::Span<std::size_t const> d_sorted_idx;

  linalg::MatrixView<float const> labels;
  common::Span<float const> predts;
  common::Span<GradientPair> gpairs;

  linalg::VectorView<GradientPair const> d_roundings;
  double const *d_cost_rounding;

  common::Span<std::size_t const> d_y_sorted_idx;

  std::int32_t iter;
};
/**
 * \brief Functor for generating pairs
 */
template <bool has_truncation>
struct MakePairsOp {
  KernelInputs args;
  /**
   * \brief Make pair for the topk pair method.
   */
  XGBOOST_DEVICE std::tuple<std::size_t, std::size_t> WithTruncation(std::size_t idx,
                                                                     bst_group_t g) const {
    auto thread_group_begin = args.d_threads_group_ptr[g];
    auto idx_in_thread_group = idx - thread_group_begin;

    auto data_group_begin = static_cast<std::size_t>(args.d_group_ptr[g]);
    std::size_t n_data = args.d_group_ptr[g + 1] - data_group_begin;
    // obtain group segment data.
    auto g_label = args.labels.Slice(linalg::Range(data_group_begin, data_group_begin + n_data), 0);
    auto g_sorted_idx = args.d_sorted_idx.subspan(data_group_begin, n_data);

    std::size_t i = 0, j = 0;
    common::UnravelTrapeziodIdx(idx_in_thread_group, n_data, &i, &j);

    std::size_t rank_high = i, rank_low = j;
    return std::make_tuple(rank_high, rank_low);
  }
  /**
   * \brief Make pair for the mean pair method
   */
  XGBOOST_DEVICE std::tuple<std::size_t, std::size_t> WithSampling(std::size_t idx,
                                                                   bst_group_t g) const {
    std::size_t n_samples = args.labels.Size();
    assert(n_samples == args.predts.size());
    // Constructed from ranking cache.
    std::size_t n_pairs =
        ltr::cuda_impl::PairsForGroup(args.d_threads_group_ptr[g + 1] - args.d_threads_group_ptr[g],
                                      args.d_group_ptr[g + 1] - args.d_group_ptr[g]);

    assert(n_pairs > 0);
    auto [sample_idx, sample_pair_idx] = linalg::UnravelIndex(idx, {n_samples, n_pairs});

    auto g_begin = static_cast<std::size_t>(args.d_group_ptr[g]);
    std::size_t n_data = args.d_group_ptr[g + 1] - g_begin;

    auto g_label = args.labels.Slice(linalg::Range(g_begin, g_begin + n_data));
    auto g_rank_idx = args.d_sorted_idx.subspan(args.d_group_ptr[g], n_data);
    auto g_y_sorted_idx = args.d_y_sorted_idx.subspan(g_begin, n_data);

    std::size_t const i = sample_idx - g_begin;
    assert(sample_pair_idx < n_samples);
    assert(i <= sample_idx);

    auto g_sorted_label = dh::MakeTransformIterator<float>(
        thrust::make_counting_iterator(0ul),
        [&](std::size_t i) { return g_label(g_rank_idx[g_y_sorted_idx[i]]); });

    // Are the labels diverse enough? If they are all the same, then there is nothing to pick
    // from another group - bail sooner
    if (g_label.Size() == 0 || g_sorted_label[0] == g_sorted_label[n_data - 1]) {
      auto z = static_cast<std::size_t>(0ul);
      return std::make_tuple(z, z);
    }

    std::size_t n_lefts = CountNumItemsToTheLeftOf(g_sorted_label, i + 1, g_sorted_label[i]);
    std::size_t n_rights =
        CountNumItemsToTheRightOf(g_sorted_label + i, n_data - i, g_sorted_label[i]);
    // The index pointing to the first element of the next bucket
    std::size_t right_bound = n_data - n_rights;

    thrust::minstd_rand rng(args.iter);
    auto pair_idx = i;
    rng.discard(sample_pair_idx * n_data + g + pair_idx);  // fixme
    thrust::uniform_int_distribution<std::size_t> dist(0, n_lefts + n_rights - 1);
    auto ridx = dist(rng);
    SPAN_CHECK(ridx < n_lefts + n_rights);
    if (ridx >= n_lefts) {
      ridx = ridx - n_lefts + right_bound;  // fixme
    }

    auto idx0 = g_y_sorted_idx[pair_idx];
    auto idx1 = g_y_sorted_idx[ridx];

    return std::make_tuple(idx0, idx1);
  }
  /**
   * \brief Generate a single pair.
   *
   * \param idx Pair index (CUDA thread index).
   * \param g   Query group index.
   */
  XGBOOST_DEVICE auto operator()(std::size_t idx, bst_group_t g) const {
    if (has_truncation) {
      return this->WithTruncation(idx, g);
    } else {
      return this->WithSampling(idx, g);
    }
  }
};
}  // namespace xgboost::obj::cuda_impl
#endif  // XGBOOST_OBJECTIVE_LAMBDARANK_OBJ_CUH_
