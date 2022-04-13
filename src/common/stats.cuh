/*!
 * Copyright 2022 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_STATS_CUH_
#define XGBOOST_COMMON_STATS_CUH_

#include <thrust/sort.h>

#include <cstddef>

#include "device_helpers.cuh"
#include "linalg_op.cuh"
#include "xgboost/generic_parameters.h"
#include "xgboost/linalg.h"
#include "xgboost/tree_model.h"

namespace xgboost {
namespace common {
namespace detail {
inline void ResidulesPredtY(Context const* ctx, linalg::TensorView<float const, 2> d_labels,
                            common::Span<float const> d_predt,
                            linalg::Tensor<float, 2>* p_residue) {
  linalg::Tensor<float, 2>& residue = *p_residue;
  residue.SetDevice(ctx->gpu_id);
  residue.Reshape(d_labels.Shape(0), d_labels.Shape(1));

  auto d_residue = residue.View(ctx->gpu_id);
  CHECK_EQ(d_predt.size(), d_labels.Size());
  linalg::ElementWiseKernel(ctx, d_labels, [=] XGBOOST_DEVICE(size_t i, float y) mutable {
    auto idx = linalg::UnravelIndex(i, d_labels.Shape());
    size_t sample_id = std::get<0>(idx);
    size_t target_id = std::get<1>(idx);
    d_residue(sample_id, target_id) = y - d_predt[i];
  });
}

template <typename Tup>
inline void SortLeafRows(linalg::TensorView<float const, 2> d_residue,
                         RowIndexCache const& row_index, dh::device_vector<size_t>* p_sorted_idx,
                         dh::device_vector<Tup>* p_keys) {
  auto& sorted_idx = *p_sorted_idx;
  sorted_idx.resize(d_residue.Shape(0));
  dh::Iota(dh::ToSpan(sorted_idx));

  auto d_leaf_ptr = row_index.node_ptr.ConstDeviceSpan();
  auto d_row_index = row_index.row_index.ConstDeviceSpan();
  auto key_it = dh::MakeTransformIterator<Tup>(
      thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(size_t i) -> Tup {
        auto idx = linalg::UnravelIndex(i, d_residue.Shape());
        size_t sample_id = std::get<0>(idx);
        size_t target_id = std::get<1>(idx);
        auto leaf_idx = dh::SegmentId(d_leaf_ptr, sample_id);
        auto residue = d_residue(d_row_index[sample_id], target_id);
        return thrust::make_tuple(leaf_idx, residue);
      });
  dh::device_vector<Tup>& keys = *p_keys;
  keys.resize(d_residue.Size());
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

inline void SegmentedPercentile(Context const* ctx, double alpha, RowIndexCache const& row_index,
                                MetaInfo const& info, HostDeviceVector<float> const& predt,
                                HostDeviceVector<float>* quantiles) {
  CHECK(alpha >= 0 && alpha <= 1);

  auto d_predt = predt.ConstDeviceSpan();
  auto d_labels = info.labels.View(ctx->gpu_id);
  linalg::Tensor<float, 2> residue;
  detail::ResidulesPredtY(ctx, d_labels, d_predt, &residue);
  auto d_residue = residue.View(ctx->gpu_id);

  dh::device_vector<size_t> sorted_idx;
  using Tup = thrust::tuple<size_t, float>;
  dh::device_vector<Tup> keys;
  detail::SortLeafRows(d_residue, row_index, &sorted_idx, &keys);

  quantiles->SetDevice(ctx->gpu_id);
  quantiles->Resize(row_index.node_idx.Size());
  auto d_results = quantiles->DeviceSpan();
  auto d_leaf_ptr = row_index.node_ptr.ConstDeviceSpan();
  auto d_row_index = row_index.row_index.ConstDeviceSpan();
  auto d_sorted_idx = dh::ToSpan(sorted_idx);

  dh::LaunchN(row_index.node_idx.Size(), [=] XGBOOST_DEVICE(size_t i) {
    size_t target_id = 0;
    // each segment is the index of a leaf.
    size_t seg_idx = i;
    size_t begin = d_leaf_ptr[seg_idx];
    auto n = static_cast<double>(d_leaf_ptr[seg_idx + 1] - begin);

    if (alpha <= (1 / (n + 1))) {
      d_results[i] = d_residue(d_row_index[d_sorted_idx[begin]]);
      return;
    }
    if (alpha >= (n / (n + 1))) {
      d_results[i] = d_residue(d_row_index[d_sorted_idx[common::LastOf(seg_idx, d_leaf_ptr)]]);
      return;
    }

    double x = alpha * static_cast<double>(n + 1);
    double k = std::floor(x) - 1;
    double d = (x - 1) - k;
    auto v0 = d_residue(d_row_index[d_sorted_idx[begin + static_cast<size_t>(k)]], target_id);
    auto v1 = d_residue(d_row_index[d_sorted_idx[begin + static_cast<size_t>(k) + 1]], target_id);
    d_results[seg_idx] = v0 + d * (v1 - v0);
  });
}

inline void SegmentedWeightedQuantile(Context const* ctx, double alpha,
                                      RowIndexCache const& row_index, MetaInfo const& info,
                                      HostDeviceVector<float> const& predt,
                                      HostDeviceVector<float>* quantiles) {
  CHECK(alpha >= 0 && alpha <= 1);
  auto d_predt = predt.ConstDeviceSpan();
  auto d_labels = info.labels.View(ctx->gpu_id);
  linalg::Tensor<float, 2> residue{d_labels.Shape(), ctx->gpu_id};
  detail::ResidulesPredtY(ctx, d_labels, d_predt, &residue);
  auto d_residue = residue.View(ctx->gpu_id);

  dh::device_vector<size_t> sorted_idx;
  using Tup = thrust::tuple<size_t, float>;
  dh::device_vector<Tup> keys;
  detail::SortLeafRows(d_residue, row_index, &sorted_idx, &keys);
  auto d_sorted_idx = dh::ToSpan(sorted_idx);
  auto d_keys = dh::ToSpan(keys);

  auto weights = info.weights_.ConstDeviceSpan();

  dh::device_vector<float> weights_cdf(weights.size());
  CHECK_EQ(weights.size(), d_labels.Shape(0));

  dh::XGBCachingDeviceAllocator<char> caching;
  Span<size_t const> node_ptr = row_index.node_ptr.ConstDeviceSpan();
  // fixme: avoid this binary search by reusing key
  auto scan_key = dh::MakeTransformIterator<size_t>(
      thrust::make_counting_iterator(0ul),
      [=] XGBOOST_DEVICE(size_t i) { return thrust::get<0>(d_keys[i]); });
  auto scan_val = dh::MakeTransformIterator<float>(
      thrust::make_counting_iterator(0ul),
      [=] XGBOOST_DEVICE(size_t i) { return weights[d_sorted_idx[i]]; });
  thrust::inclusive_scan_by_key(thrust::cuda::par(caching), scan_key, scan_key + weights.size(),
                                dh::tcbegin(weights), weights_cdf.begin());

  quantiles->SetDevice(ctx->gpu_id);
  quantiles->Resize(row_index.node_idx.Size());
  auto d_results = quantiles->DeviceSpan();
  auto d_leaf_ptr = row_index.node_ptr.ConstDeviceSpan();
  auto d_row_index = row_index.row_index.ConstDeviceSpan();
  auto d_weight_cdf = dh::ToSpan(weights_cdf);

  dh::LaunchN(row_index.node_idx.Size(), [=] XGBOOST_DEVICE(size_t i) {
    size_t target_id = 0;
    size_t seg_idx = i;
    size_t begin = d_leaf_ptr[seg_idx];
    auto n = static_cast<double>(d_leaf_ptr[seg_idx + 1] - begin);
    auto leaf_cdf = d_weight_cdf.subspan(begin, n);
    float thresh = leaf_cdf.back() * alpha;

    size_t idx = thrust::lower_bound(thrust::seq, leaf_cdf.data(),
                                     leaf_cdf.data() + leaf_cdf.size(), thresh) -
                 leaf_cdf.data();
    idx = std::min(idx, static_cast<size_t>(n - 1));
    if (idx == 0 || idx == static_cast<size_t>(n - 1)) {
      d_results[i] = d_residue(d_row_index[d_sorted_idx[idx + begin]], target_id);
      return;
    }

    float v0 = d_residue(d_row_index[d_sorted_idx[idx + begin]], target_id);
    float v1 = d_residue(d_row_index[d_sorted_idx[idx + begin + 1]], target_id);

    if (leaf_cdf[idx + 1] - leaf_cdf[idx] >= 1.0f) {
      auto v = (thresh - leaf_cdf[idx]) / (leaf_cdf[idx + 1] - leaf_cdf[idx]) * (v1 - v1) + v0;
      d_results[i] = v;
    } else {
      d_results[i] = v1;
    }
  });
}
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_STATS_CUH_
