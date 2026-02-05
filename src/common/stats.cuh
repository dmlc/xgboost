/**
 * Copyright 2022-2026, XGBoost Contributors
 */
#pragma once

#include <thrust/binary_search.h>                  // for lower_bound
#include <thrust/for_each.h>                       // for for_each_n
#include <thrust/iterator/constant_iterator.h>     // for make_constant_iterator
#include <thrust/iterator/counting_iterator.h>     // for make_counting_iterator
#include <thrust/iterator/permutation_iterator.h>  // for make_permutation_iterator
#include <thrust/scan.h>                           // for inclusive_scan_by_key

#include <algorithm>    // for min
#include <cstddef>      // for size_t
#include <iterator>     // for distance
#include <limits>       // for numeric_limits
#include <type_traits>  // for is_floating_point_v,iterator_traits

#include "algorithm.cuh"       // for SegmentedArgMergeSort
#include "cuda_context.cuh"    // for CUDAContext
#include "device_helpers.cuh"  // for SegmentId
#include "device_vector.cuh"   // for device_vector
#include "xgboost/context.h"   // for Context
#include "xgboost/linalg.h"    // for UnravelIndex
#include "xgboost/span.h"      // for Span

namespace xgboost::common {
namespace detail {
// This should be a lambda function, but for some reason gcc-11 + nvcc-11.8 failed to
// compile it. As a result, a functor is extracted instead.
//
// error: ‘__T288’ was not declared in this scope
template <typename SegIt, typename ValIt, typename AlphaIt>
struct QuantileSegmentOp {
  SegIt seg_begin;
  ValIt val;
  AlphaIt alpha_it;
  linalg::VectorView<float> d_results;

  static_assert(std::is_floating_point_v<typename std::iterator_traits<ValIt>::value_type>,
                "Invalid value for quantile.");
  static_assert(std::is_floating_point_v<typename std::iterator_traits<ValIt>::value_type>,
                "Invalid alpha.");

  XGBOOST_DEVICE void operator()(std::size_t seg_idx) {
    std::size_t begin = seg_begin[seg_idx];
    auto n = static_cast<double>(seg_begin[seg_idx + 1] - begin);
    double a = alpha_it[seg_idx];

    if (n == 0) {
      d_results(seg_idx) = std::numeric_limits<float>::quiet_NaN();
      return;
    }

    if (a <= (1 / (n + 1))) {
      d_results(seg_idx) = val[begin];
      return;
    }
    if (a >= (n / (n + 1))) {
      d_results(seg_idx) = val[common::LastOf(seg_idx, seg_begin)];
      return;
    }

    double x = a * static_cast<double>(n + 1);
    double k = std::floor(x) - 1;
    double d = (x - 1) - k;

    auto v0 = val[begin + static_cast<std::size_t>(k)];
    auto v1 = val[begin + static_cast<std::size_t>(k) + 1];

    d_results(seg_idx) = v0 + d * (v1 - v0);
  }
};

template <typename SegIt, typename ValIt, typename AlphaIt>
XGBOOST_DEVICE auto MakeQSegOp(SegIt seg_it, ValIt val_it, AlphaIt alpha_it,
                               linalg::VectorView<float> d_results) {
  return QuantileSegmentOp<SegIt, ValIt, AlphaIt>{seg_it, val_it, alpha_it, d_results};
}

template <typename SegIt>
struct SegOp {
  SegIt seg_beg;
  SegIt seg_end;

  XGBOOST_DEVICE std::size_t operator()(std::size_t i) {
    return dh::SegmentId(seg_beg, seg_end, i);
  }
};

template <typename WIter>
struct WeightOp {
  WIter w_begin;
  Span<std::size_t const> d_sorted_idx;
  XGBOOST_DEVICE float operator()(std::size_t i) { return w_begin[d_sorted_idx[i]]; }
};

template <typename SegIt, typename ValIt, typename AlphaIt>
struct WeightedQuantileSegOp {
  AlphaIt alpha_it;
  SegIt seg_beg;
  ValIt val_begin;
  Span<float const> d_weight_cdf;
  Span<std::size_t const> d_sorted_idx;
  linalg::VectorView<float> d_results;
  static_assert(std::is_floating_point_v<typename std::iterator_traits<AlphaIt>::value_type>,
                "Invalid alpha.");
  static_assert(std::is_floating_point_v<typename std::iterator_traits<ValIt>::value_type>,
                "Invalid value for quantile.");

  XGBOOST_DEVICE void operator()(std::size_t seg_idx) {
    std::size_t begin = seg_beg[seg_idx];
    auto n = static_cast<double>(seg_beg[seg_idx + 1] - begin);
    if (n == 0) {
      d_results(seg_idx) = std::numeric_limits<float>::quiet_NaN();
      return;
    }
    auto seg_cdf = d_weight_cdf.subspan(begin, static_cast<std::size_t>(n));
    auto seg_sorted_idx = d_sorted_idx.subspan(begin, static_cast<std::size_t>(n));
    double a = alpha_it[seg_idx];
    double thresh = seg_cdf.back() * a;

    std::size_t idx =
        thrust::lower_bound(thrust::seq, seg_cdf.data(), seg_cdf.data() + seg_cdf.size(), thresh) -
        seg_cdf.data();
    idx = std::min(idx, static_cast<std::size_t>(n - 1));
    d_results(seg_idx) = val_begin[seg_sorted_idx[idx]];
  }
};

template <typename SegIt, typename ValIt, typename AlphaIt>
XGBOOST_DEVICE auto MakeWQSegOp(SegIt seg_it, ValIt val_it, AlphaIt alpha_it,
                                Span<float const> d_weight_cdf,
                                Span<std::size_t const> d_sorted_idx,
                                linalg::VectorView<float> d_results) {
  return WeightedQuantileSegOp<SegIt, ValIt, AlphaIt>{alpha_it,     seg_it,       val_it,
                                                      d_weight_cdf, d_sorted_idx, d_results};
}
}  // namespace detail
/**
 * @brief Compute segmented quantile on GPU.
 *
 * @tparam SegIt Iterator for CSR style segments indptr
 * @tparam ValIt Iterator for values
 * @tparam AlphaIt Iterator to alphas
 *
 * @param alpha The p^th quantile we want to compute, one for each segment.
 *
 *    std::distance(seg_begin, seg_end) should be equal to n_segments + 1
 */
template <typename SegIt, typename ValIt, typename AlphaIt,
          std::enable_if_t<!std::is_floating_point_v<AlphaIt>>* = nullptr>
void SegmentedQuantile(Context const* ctx, AlphaIt alpha_it, SegIt seg_begin, SegIt seg_end,
                       ValIt val_begin, ValIt val_end, HostDeviceVector<float>* quantiles) {
  dh::device_vector<std::size_t> sorted_idx;
  common::SegmentedArgMergeSort(ctx, seg_begin, seg_end, val_begin, val_end, &sorted_idx);
  auto n_segments = std::distance(seg_begin, seg_end) - 1;
  if (n_segments <= 0) {
    return;
  }

  auto d_sorted_idx = dh::ToSpan(sorted_idx);
  auto val = thrust::make_permutation_iterator(val_begin, dh::tcbegin(d_sorted_idx));

  quantiles->SetDevice(ctx->Device());
  quantiles->Resize(n_segments);
  auto d_results = linalg::MakeVec(ctx->Device(), quantiles->DeviceSpan());

  thrust::for_each_n(ctx->CUDACtx()->CTP(), thrust::make_counting_iterator(0ul), n_segments,
                     detail::MakeQSegOp(seg_begin, val, alpha_it, d_results));
}

/**
 * @brief Calculate multiple quantiles for multiple segments.
 *
 *    Each segment has `n_alphas` quantiles. All segments share the same set of quantiles.
 *
 *    The output quantiles are stored in a row-major matrix with shape: (n_segments, n_alphas).
 *
 * @param h_alphas Quantiles to be estimated.
 * @param values   A callable object that indexes the value matrix with shape (n, n_alphas).
 * @param n        The number of samples in values matrix, should equal to *(seg_end - 1).
 */
template <typename SegIt, typename ValIt>
void SegmentedQuantile(Context const* ctx, std::vector<float> const& h_alphas, SegIt seg_begin,
                       SegIt seg_end, ValIt values, std::size_t n,
                       HostDeviceVector<float>* quantiles) {
  // The values is a matrix with shape (n_samples, n_alphas), we have a 2-way segment
  // here.
  // For now, we simply iterate through the alphas on host as we are likely to have at most 3
  // quantiles to compute.
  auto n_segments = std::distance(seg_begin, seg_end) - 1;
  if (n_segments <= 0) {
    return;
  }

  auto n_alphas = h_alphas.size();
  quantiles->SetDevice(ctx->Device());
  quantiles->Resize(n_segments * n_alphas);
  auto d_quantiles = linalg::MakeTensorView(ctx, quantiles->DeviceSpan(), n_segments, n_alphas);

  for (std::size_t alpha_idx = 0; alpha_idx < n_alphas; ++alpha_idx) {
    auto val_begin = dh::MakeIndexTransformIter(
        [=] XGBOOST_DEVICE(std::size_t i) { return values(i, alpha_idx); });
    auto val_end = val_begin + n;
    dh::device_vector<std::size_t> sorted_idx;
    common::SegmentedArgMergeSort(ctx, seg_begin, seg_end, val_begin, val_end, &sorted_idx);

    auto d_sorted_idx = dh::ToSpan(sorted_idx);
    auto val = thrust::make_permutation_iterator(val_begin, dh::tcbegin(d_sorted_idx));

    auto row = d_quantiles.Slice(linalg::All(), alpha_idx);
    auto alpha = h_alphas[alpha_idx];
    thrust::for_each_n(
        ctx->CUDACtx()->CTP(), thrust::make_counting_iterator(0ul), n_segments,
        detail::MakeQSegOp(seg_begin, val, thrust::make_constant_iterator(alpha), row));
  }
}

/**
 * @brief Compute segmented quantile on GPU.
 *
 * @tparam SegIt Iterator for CSR style segments indptr
 * @tparam ValIt Iterator for values
 *
 * @param alpha The p^th quantile we want to compute
 *
 *    std::distance(ptr_begin, ptr_end) should be equal to n_segments + 1
 */
template <typename SegIt, typename ValIt>
void SegmentedQuantile(Context const* ctx, double alpha, SegIt seg_begin, SegIt seg_end,
                       ValIt val_begin, ValIt val_end, HostDeviceVector<float>* quantiles) {
  CHECK(alpha >= 0 && alpha <= 1);
  auto alpha_it = thrust::make_constant_iterator(alpha);
  return SegmentedQuantile(ctx, alpha_it, seg_begin, seg_end, val_begin, val_end, quantiles);
}

/**
 * @brief Compute segmented quantile on GPU with weighted inputs.
 *
 * @tparam SegIt Iterator for CSR style segments indptr
 * @tparam ValIt Iterator for values
 * @tparam WIter Iterator for weights
 *
 * @param alpha_it Iterator for the p^th quantile we want to compute, one per-segment
 * @param w_begin  Iterator for weight for each input element
 */
template <typename SegIt, typename ValIt, typename AlphaIt, typename WIter,
          typename std::enable_if_t<
              !std::is_same_v<typename std::iterator_traits<AlphaIt>::value_type, void>>* = nullptr>
void SegmentedWeightedQuantile(Context const* ctx, AlphaIt alpha_it, SegIt seg_beg, SegIt seg_end,
                               ValIt val_begin, ValIt val_end, WIter w_begin, WIter w_end,
                               HostDeviceVector<float>* quantiles) {
  auto cuctx = ctx->CUDACtx();
  dh::device_vector<std::size_t> sorted_idx;
  common::SegmentedArgMergeSort(ctx, seg_beg, seg_end, val_begin, val_end, &sorted_idx);
  auto d_sorted_idx = dh::ToSpan(sorted_idx);
  std::size_t n_weights = std::distance(w_begin, w_end);
  dh::device_vector<float> weights_cdf(n_weights);
  std::size_t n_elems = std::distance(val_begin, val_end);
  CHECK_EQ(n_weights, n_elems);

  auto scan_key = dh::MakeIndexTransformIter(detail::SegOp<SegIt>{seg_beg, seg_end});
  auto scan_val = dh::MakeIndexTransformIter(detail::WeightOp<WIter>{w_begin, d_sorted_idx});
  thrust::inclusive_scan_by_key(cuctx->CTP(), scan_key, scan_key + n_weights, scan_val,
                                weights_cdf.begin());

  auto n_segments = std::distance(seg_beg, seg_end) - 1;
  quantiles->SetDevice(ctx->Device());
  quantiles->Resize(n_segments);
  auto d_results = linalg::MakeVec(ctx->Device(), quantiles->DeviceSpan());
  auto d_weight_cdf = dh::ToSpan(weights_cdf);

  thrust::for_each_n(
      cuctx->CTP(), thrust::make_counting_iterator(0ul), n_segments,
      detail::MakeWQSegOp(seg_beg, val_begin, alpha_it, d_weight_cdf, d_sorted_idx, d_results));
}

/**
 * @brief Calculate multiple weighted quantiles for multiple segments.
 *
 * @param h_alphas Quantiles to be estimated.
 * @param values   A callable object that indexes the value matrix with shape (n, n_alphas).
 * @param n        The number of samples in values matrix, should equal to *(seg_end - 1).
 */
template <typename SegIt, typename ValIt, typename WIter>
void SegmentedWeightedQuantile(Context const* ctx, std::vector<float> const& h_alphas,
                               SegIt seg_beg, SegIt seg_end, ValIt values, WIter w_begin,
                               WIter w_end, HostDeviceVector<float>* quantiles) {
  auto cuctx = ctx->CUDACtx();

  auto n_segments = std::distance(seg_beg, seg_end) - 1;
  if (n_segments <= 0) {
    return;
  }
  auto n_alphas = h_alphas.size();
  std::size_t n = std::distance(w_begin, w_end);

  quantiles->SetDevice(ctx->Device());
  quantiles->Resize(n_segments * n_alphas);
  auto d_quantiles = linalg::MakeTensorView(ctx, quantiles->DeviceSpan(), n_segments, n_alphas);

  for (std::size_t alpha_idx = 0; alpha_idx < n_alphas; ++alpha_idx) {
    auto val_begin = dh::MakeIndexTransformIter(
        [=] XGBOOST_DEVICE(std::size_t i) { return values(i, alpha_idx); });
    auto val_end = val_begin + n;

    dh::device_vector<std::size_t> sorted_idx;
    common::SegmentedArgMergeSort(ctx, seg_beg, seg_end, val_begin, val_end, &sorted_idx);

    auto d_sorted_idx = dh::ToSpan(sorted_idx);
    dh::device_vector<float> weights_cdf(n);

    auto scan_key = dh::MakeIndexTransformIter(detail::SegOp<SegIt>{seg_beg, seg_end});
    auto scan_val = dh::MakeIndexTransformIter(detail::WeightOp<WIter>{w_begin, d_sorted_idx});
    thrust::inclusive_scan_by_key(cuctx->CTP(), scan_key, scan_key + n, scan_val,
                                  weights_cdf.begin());
    auto d_weight_cdf = dh::ToSpan(weights_cdf);
    auto alpha = h_alphas[alpha_idx];
    auto row = d_quantiles.Slice(linalg::All(), alpha_idx);

    thrust::for_each_n(
        cuctx->CTP(), thrust::make_counting_iterator(0ul), n_segments,
        detail::MakeWQSegOp(seg_beg, val_begin, thrust::make_constant_iterator(alpha), d_weight_cdf,
                            d_sorted_idx, row));
  }
}

template <typename SegIt, typename ValIt, typename WIter>
void SegmentedWeightedQuantile(Context const* ctx, double alpha, SegIt seg_beg, SegIt seg_end,
                               ValIt val_begin, ValIt val_end, WIter w_begin, WIter w_end,
                               HostDeviceVector<float>* quantiles) {
  CHECK(alpha >= 0 && alpha <= 1);
  return SegmentedWeightedQuantile(ctx, thrust::make_constant_iterator(alpha), seg_beg, seg_end,
                                   val_begin, val_end, w_begin, w_end, quantiles);
}
}  // namespace xgboost::common
