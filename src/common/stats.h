/*!
 * Copyright 2022 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_STATS_H_
#define XGBOOST_COMMON_STATS_H_
#include <algorithm>
#include <iterator>
#include <limits>
#include <vector>

#include "common.h"  // AssertGPUSupport
#include "xgboost/generic_parameters.h"
#include "xgboost/linalg.h"

namespace xgboost {
namespace common {

/**
 * \brief Percentile with masked array using linear interpolation.
 *
 *   https://www.itl.nist.gov/div898/handbook/prc/section2/prc262.htm
 *
 * \param alpha Percentile, must be in range [0, 1].
 * \param begin Iterator begin for input array.
 * \param end   Iterator end for input array.
 *
 * \return The result of interpolation.
 */
template <typename Iter>
float Quantile(double alpha, Iter const& begin, Iter const& end) {
  CHECK(alpha >= 0 && alpha <= 1);
  auto n = static_cast<double>(std::distance(begin, end));
  if (n == 0) {
    return std::numeric_limits<float>::quiet_NaN();
  }

  std::vector<size_t> sorted_idx(n);
  std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
  std::stable_sort(sorted_idx.begin(), sorted_idx.end(),
                   [&](size_t l, size_t r) { return *(begin + l) < *(begin + r); });

  auto val = [&](size_t i) { return *(begin + sorted_idx[i]); };
  static_assert(std::is_same<decltype(val(0)), float>::value, "");

  if (alpha <= (1 / (n + 1))) {
    return val(0);
  }
  if (alpha >= (n / (n + 1))) {
    return val(sorted_idx.size() - 1);
  }
  assert(n != 0 && "The number of rows in a leaf can not be zero.");
  double x = alpha * static_cast<double>((n + 1));
  double k = std::floor(x) - 1;
  CHECK_GE(k, 0);
  double d = (x - 1) - k;

  auto v0 = val(static_cast<size_t>(k));
  auto v1 = val(static_cast<size_t>(k) + 1);
  return v0 + d * (v1 - v0);
}

/**
 * \brief Calculate the weighted quantile with step function. Unlike the unweighted
 *        version, no interpolation is used.
 *
 *   See https://aakinshin.net/posts/weighted-quantiles/ for some discussion on computing
 *   weighted quantile with interpolation.
 */
template <typename Iter, typename WeightIter>
float WeightedQuantile(double alpha, Iter begin, Iter end, WeightIter weights) {
  auto n = static_cast<double>(std::distance(begin, end));
  if (n == 0) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  std::vector<size_t> sorted_idx(n);
  std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
  std::stable_sort(sorted_idx.begin(), sorted_idx.end(),
                   [&](size_t l, size_t r) { return *(begin + l) < *(begin + r); });

  auto val = [&](size_t i) { return *(begin + sorted_idx[i]); };

  std::vector<float> weight_cdf(n);  // S_n
  // weighted cdf is sorted during construction
  weight_cdf[0] = *(weights + sorted_idx[0]);
  for (size_t i = 1; i < n; ++i) {
    weight_cdf[i] = weight_cdf[i - 1] + *(weights + sorted_idx[i]);
  }
  float thresh = weight_cdf.back() * alpha;
  size_t idx =
      std::lower_bound(weight_cdf.cbegin(), weight_cdf.cend(), thresh) - weight_cdf.cbegin();
  idx = std::min(idx, static_cast<size_t>(n - 1));
  return val(idx);
}

namespace cuda {
float Median(Context const* ctx, linalg::TensorView<float const, 2> t,
             common::OptionalWeights weights);
#if !defined(XGBOOST_USE_CUDA)
inline float Median(Context const*, linalg::TensorView<float const, 2>, common::OptionalWeights) {
  common::AssertGPUSupport();
  return 0;
}
#endif  // !defined(XGBOOST_USE_CUDA)

float Mean(Context const* ctx, linalg::TensorView<float const, 2> t,
           common::OptionalWeights weights, size_t n);

#if !defined(XGBOOST_USE_CUDA)
inline float Mean(Context const*, linalg::TensorView<float const, 2>, common::OptionalWeights,
                  size_t n) {
  AssertGPUSupport();
  return 0;
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace cuda

inline float Median(Context const* ctx, linalg::Tensor<float, 2> const& t,
                    HostDeviceVector<float> const& weights) {
  CHECK_EQ(t.Shape(1), 0) << "Matrix is not yet supported.";
  if (!ctx->IsCPU()) {
    weights.SetDevice(ctx->gpu_id);
    auto opt_weights = OptionalWeights(weights.ConstDeviceSpan());
    auto t_v = t.View(ctx->gpu_id);
    return cuda::Median(ctx, t_v, opt_weights);
  }

  auto opt_weights = OptionalWeights(weights.ConstHostSpan());
  auto t_v = t.HostView();
  auto iter = common::MakeIndexTransformIter(
      [&](size_t i) { return linalg::detail::Apply(t_v, linalg::UnravelIndex(i, t_v.Shape())); });
  float q{0};
  if (opt_weights.Empty()) {
    q = common::Quantile(0.5, iter, iter + t_v.Size());
  } else {
    CHECK_NE(t_v.Shape(1), 0);
    auto w_it = common::MakeIndexTransformIter([&](size_t i) {
      auto sample_idx = i / t_v.Shape(1);
      return opt_weights[sample_idx];
    });
    q = common::WeightedQuantile(0.5, iter, iter + t_v.Size(), w_it);
  }
  return q;
}

/**
 * \brief Calculate mean or partial mean. When n is specified to be non-zero, we use n as
 *        the total number of elements instead of the size of t.
 */
inline float Mean(Context const* ctx, linalg::Tensor<float, 2> const& t,
                  HostDeviceVector<float> const& weights, size_t n = 0) {
  if (!weights.Empty()) {
    CHECK_EQ(weights.Size(), t.Shape(0)) << "Weight is assigned for each row.";
  }
  if (!ctx->IsCPU()) {
    weights.SetDevice(ctx->gpu_id);
    auto opt_weights = OptionalWeights(weights.ConstDeviceSpan());
    auto t_v = t.View(ctx->gpu_id);
    cuda::Mean(ctx, t_v, opt_weights, n);
  }

  auto opt_weights = OptionalWeights(weights.ConstHostSpan());
  auto t_v = t.HostView();

  MemStackAllocator<float, 128> mean_tloc(ctx->Threads(), 0.0f);
  auto iter = common::MakeIndexTransformIter(
      [&](size_t i) { return linalg::detail::Apply(t_v, linalg::UnravelIndex(i, t_v.Shape())); });

  double size = n == 0 ? t_v.Size() : n;
  CHECK_NE(size, 0);
  ParallelFor(t_v.Size(), ctx->Threads(), [&](auto i) {
    auto tidx = omp_get_thread_num();
    auto ridx = std::get<0>(linalg::UnravelIndex(i, t_v.Shape()));
    mean_tloc[tidx] += iter[i] * opt_weights[ridx] / size;
  });
  auto mean = std::accumulate(mean_tloc.cbegin(), mean_tloc.cend(), 0.0f);
  return mean;
}
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_STATS_H_
