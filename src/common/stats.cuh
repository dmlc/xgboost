/*!
 * Copyright 2022 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_STATS_CUH_
#define XGBOOST_COMMON_STATS_CUH_

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/sort.h>

#include <cstddef>
#include <iterator>  // std::distance

#include "device_helpers.cuh"
#include "linalg_op.cuh"
#include "xgboost/generic_parameters.h"
#include "xgboost/linalg.h"
#include "xgboost/tree_model.h"

namespace xgboost {
namespace common {
namespace detail {
template <typename SegIt, typename ValIt>
inline void SegmentedArgSort(SegIt seg_begin, SegIt seg_end, ValIt val_begin, ValIt val_end,
                             dh::device_vector<size_t>* p_sorted_idx) {
  using Tup = thrust::tuple<size_t, float>;
  auto& sorted_idx = *p_sorted_idx;
  size_t n = std::distance(val_begin, val_end);
  sorted_idx.resize(n);
  dh::Iota(dh::ToSpan(sorted_idx));
  dh::device_vector<Tup> keys(sorted_idx.size());
  auto key_it = dh::MakeTransformIterator<Tup>(
      thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(size_t i) -> Tup {
        auto leaf_idx = dh::SegmentId(seg_begin, seg_end, i);
        auto residue = val_begin[i];
        return thrust::make_tuple(leaf_idx, residue);
      });
  dh::XGBCachingDeviceAllocator<char> caching;
  thrust::copy(thrust::cuda::par(caching), key_it, key_it + keys.size(), keys.begin());

  dh::XGBDeviceAllocator<char> alloc;
  thrust::stable_sort_by_key(thrust::cuda::par(alloc), keys.begin(), keys.end(), sorted_idx.begin(),
                             [=] XGBOOST_DEVICE(Tup const& l, Tup const& r) {
                               if (thrust::get<0>(l) != thrust::get<0>(r)) {
                                 return thrust::get<0>(l) < thrust::get<0>(r);  // segment index
                               }
                               return thrust::get<1>(l) < thrust::get<1>(r);  // residue
                             });
}
}  // namespace detail

/**
 * \brief Compute segmented quantile on GPU.
 *
 * \tparam SegIt Iterator for CSR style segments indptr
 * \tparam ValIt Iterator for values
 *
 * \param alpha The p^th quantile we want to compute
 *
 *    std::distance(ptr_begin, ptr_end) should be equal to n_segments + 1
 */
template <typename SegIt, typename ValIt>
void SegmentedQuantile(Context const* ctx, double alpha, SegIt seg_begin, SegIt seg_end,
                       ValIt val_begin, ValIt val_end, HostDeviceVector<float>* quantiles) {
  CHECK(alpha >= 0 && alpha <= 1);

  dh::device_vector<size_t> sorted_idx;
  using Tup = thrust::tuple<size_t, float>;
  detail::SegmentedArgSort(seg_begin, seg_end, val_begin, val_end, &sorted_idx);
  auto n_segments = std::distance(seg_begin, seg_end) - 1;
  if (n_segments <= 0) {
    return;
  }

  quantiles->SetDevice(ctx->gpu_id);
  quantiles->Resize(n_segments);
  auto d_results = quantiles->DeviceSpan();
  auto d_sorted_idx = dh::ToSpan(sorted_idx);

  auto val = thrust::make_permutation_iterator(val_begin, dh::tcbegin(d_sorted_idx));

  dh::LaunchN(n_segments, [=] XGBOOST_DEVICE(size_t i) {
    // each segment is the index of a leaf.
    size_t seg_idx = i;
    size_t begin = seg_begin[seg_idx];
    auto n = static_cast<double>(seg_begin[seg_idx + 1] - begin);

    if (alpha <= (1 / (n + 1))) {
      d_results[i] = val[begin];
      return;
    }
    if (alpha >= (n / (n + 1))) {
      d_results[i] = val[common::LastOf(seg_idx, seg_begin)];
      return;
    }

    double x = alpha * static_cast<double>(n + 1);
    double k = std::floor(x) - 1;
    double d = (x - 1) - k;
    auto v0 = val[begin + static_cast<size_t>(k)];
    auto v1 = val[begin + static_cast<size_t>(k) + 1];
    d_results[seg_idx] = v0 + d * (v1 - v0);
  });
}

template <typename SegIt, typename ValIt, typename WIter>
void SegmentedWeightedQuantile(Context const* ctx, double alpha, SegIt seg_beg, SegIt seg_end,
                               ValIt val_begin, ValIt val_end, WIter w_begin, WIter w_end,
                               HostDeviceVector<float>* quantiles) {
  CHECK(alpha >= 0 && alpha <= 1);
  dh::device_vector<size_t> sorted_idx;
  detail::SegmentedArgSort(seg_beg, seg_end, val_begin, val_end, &sorted_idx);
  auto d_sorted_idx = dh::ToSpan(sorted_idx);
  size_t n_samples = std::distance(w_begin, w_end);
  dh::device_vector<float> weights_cdf(n_samples);

  dh::XGBCachingDeviceAllocator<char> caching;
  auto scan_key = dh::MakeTransformIterator<size_t>(
      thrust::make_counting_iterator(0ul),
      [=] XGBOOST_DEVICE(size_t i) { return dh::SegmentId(seg_beg, seg_end, i); });
  auto scan_val = dh::MakeTransformIterator<float>(
      thrust::make_counting_iterator(0ul),
      [=] XGBOOST_DEVICE(size_t i) { return w_begin[d_sorted_idx[i]]; });
  thrust::inclusive_scan_by_key(thrust::cuda::par(caching), scan_key, scan_key + n_samples,
                                scan_val, weights_cdf.begin());

  auto n_segments = std::distance(seg_beg, seg_end) - 1;
  quantiles->SetDevice(ctx->gpu_id);
  quantiles->Resize(n_segments);
  auto d_results = quantiles->DeviceSpan();
  auto d_weight_cdf = dh::ToSpan(weights_cdf);

  dh::LaunchN(n_segments, [=] XGBOOST_DEVICE(size_t i) {
    size_t seg_idx = i;
    size_t begin = seg_beg[seg_idx];
    auto n = static_cast<double>(seg_beg[seg_idx + 1] - begin);
    auto leaf_cdf = d_weight_cdf.subspan(begin, static_cast<size_t>(n));
    auto leaf_sorted_idx = d_sorted_idx.subspan(begin, static_cast<size_t>(n));
    float thresh = leaf_cdf.back() * alpha;

    size_t idx = thrust::lower_bound(thrust::seq, leaf_cdf.data(),
                                     leaf_cdf.data() + leaf_cdf.size(), thresh) -
                 leaf_cdf.data();
    idx = std::min(idx, static_cast<size_t>(n - 1));
    d_results[i] = val_begin[leaf_sorted_idx[idx]];
  });
}
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_STATS_CUH_
