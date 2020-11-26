/*!
 * Copyright 2020 XGBoost contributors
 */
#ifndef XGBOOST_OBJECTIVE_RANK_OBJ_CUH_
#define XGBOOST_OBJECTIVE_RANK_OBJ_CUH_

#include <cub/cub.cuh>
#include "../common/device_helpers.cuh"
#include "../common/ranking_utils.h"
#include "../common/math.h"

namespace xgboost {
namespace obj {
template <bool with_diagonal = false>
XGBOOST_DEVICE size_t TrapezoidArea(size_t n, size_t h) {
  if (!with_diagonal) {
    n -= 1;
  }
  h = std::min(n, h);  // Specific for ranking.
  size_t total = ((n - (h - 1)) + n) * h / 2;
  return total;
}

template <bool with_diagonal = false, typename U>
inline size_t
SegmentedTrapezoidThreads(xgboost::common::Span<U> group_ptr,
                          xgboost::common::Span<size_t> out_group_threads_ptr,
                          size_t h) {
  CHECK_GE(group_ptr.size(), 1);
  CHECK_EQ(group_ptr.size(), out_group_threads_ptr.size());
  dh::LaunchN(
      dh::CurrentDevice(), group_ptr.size(), [=] XGBOOST_DEVICE(size_t idx) {
        if (idx == 0) {
          out_group_threads_ptr[0] = 0;
          return;
        }

        size_t cnt = static_cast<size_t>(group_ptr[idx] - group_ptr[idx - 1]);
        out_group_threads_ptr[idx] = TrapezoidArea<with_diagonal>(cnt, h);
      });
  size_t bytes = 0;
  cub::DeviceScan::InclusiveSum(nullptr, bytes, out_group_threads_ptr.data(),
                                out_group_threads_ptr.data(),
                                out_group_threads_ptr.size());
  dh::TemporaryArray<xgboost::common::byte> temp_storage(bytes);
  cub::DeviceScan::InclusiveSum(
      temp_storage.data().get(), bytes, out_group_threads_ptr.data(),
      out_group_threads_ptr.data(), out_group_threads_ptr.size());
  size_t total = 0;
  dh::safe_cuda(cudaMemcpy(
      &total, out_group_threads_ptr.data() + out_group_threads_ptr.size() - 1,
      sizeof(total), cudaMemcpyDeviceToHost));
  return total;
}

inline void CalcQueriesInvIDCG(common::Span<float const> d_labels,
                               common::Span<bst_group_t const> group_ptr,
                               common::Span<float> out_inv_IDCG,
                               size_t truncation) {
  thrust::device_vector<float> sorted_labels(d_labels.size());
  CHECK_GE(group_ptr.size(), 1ul);
  size_t n_groups = group_ptr.size() - 1;
  CHECK_EQ(out_inv_IDCG.size(), n_groups);

  auto d_sorted_labels = dh::ToSpan(sorted_labels);
  dh::LaunchN(
      dh::CurrentDevice(), sorted_labels.size(),
      [=] XGBOOST_DEVICE(size_t idx) { d_sorted_labels[idx] = d_labels[idx]; });
  dh::SegmentedSortKeys<true>(group_ptr, dh::ToSpan(sorted_labels));

  using IdxGroup = thrust::pair<size_t, size_t>;
  auto group_it = dh::MakeTransformIterator<IdxGroup>(
      thrust::make_counting_iterator(0ull), [=] XGBOOST_DEVICE(size_t idx) {
        return thrust::make_pair(idx, dh::SegmentId(group_ptr, idx));
      });
  auto value_it = dh::MakeTransformIterator<float>(
      group_it, [d_sorted_labels, group_ptr, truncation] XGBOOST_DEVICE(IdxGroup const& l) {
        auto g_begin = group_ptr[l.second];
        auto idx_in_group = l.first - g_begin;
        if (idx_in_group >= truncation) {
          // Truncating IDCG, this wastes some threads but IDCG is calcuated once per
          // dataset so should be fine.
          return 0.0f;
        }

        auto g_labels = d_sorted_labels.subspan(g_begin, group_ptr[l.second + 1] - g_begin);
        auto label = g_labels[idx_in_group];
        auto gain = CalcNDCGGain(label);
        auto discount = CalcNDCGDiscount(idx_in_group);
        return gain * discount;
      });
  dh::XGBDeviceAllocator<common::byte> alloc;
  thrust::reduce_by_key(
      thrust::cuda::par(alloc), group_it, group_it + sorted_labels.size(),
      value_it, thrust::make_discard_iterator(), dh::tbegin(out_inv_IDCG),
      [] XGBOOST_DEVICE(IdxGroup const& l, IdxGroup const& r) {
        return l.second == r.second;
      },
      thrust::plus<float>{});
  dh::LaunchN(dh::CurrentDevice(), out_inv_IDCG.size(),
              [out_inv_IDCG] XGBOOST_DEVICE(size_t idx) {
                float idcg = out_inv_IDCG[idx];
                out_inv_IDCG[idx] = idcg == 0.0f ? 0.0f : 1.0f / idcg;
              });
}

XGBOOST_DEVICE inline void UnravelTrapeziodIdx(size_t i_idx, size_t n,
                                               size_t *out_i, size_t *out_j) {
  auto &i = *out_i;
  auto &j = *out_j;
  double idx = static_cast<double>(i_idx);
  double N = static_cast<double>(n);

  i = std::ceil(-(0.5 - N + std::sqrt(common::Sqr(N - 0.5) + 2.0 * (-idx - 1.0)))) - 1.0;

  auto I = static_cast<double>(i);
  size_t n_elems = -0.5 * common::Sqr(I) + (N - 0.5) * I;

  j = idx - n_elems + i + 1;
}
}  // namespace obj
}  // namespace xgboost
#endif  // XGBOOST_OBJECTIVE_RANK_OBJ_CUH_
