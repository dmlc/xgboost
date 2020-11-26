/*!
 * Copyright 2021 XGBoost contributors
 */
#ifndef XGBOOST_OBJECTIVE_RANK_OBJ_CUH_
#define XGBOOST_OBJECTIVE_RANK_OBJ_CUH_

#include "../common/device_helpers.cuh"
#include "../common/math.h"
#include "../common/ranking_utils.h"

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

inline void CalcQueriesInvIDCG(common::Span<float const> d_labels,
                               common::Span<bst_group_t const> group_ptr,
                               common::Span<float> out_inv_IDCG, size_t truncation) {
  thrust::device_vector<float> sorted_labels(d_labels.size());
  CHECK_GE(group_ptr.size(), 1ul);
  size_t n_groups = group_ptr.size() - 1;
  CHECK_EQ(out_inv_IDCG.size(), n_groups);

  auto d_sorted_labels = dh::ToSpan(sorted_labels);
  dh::LaunchN(sorted_labels.size(),
              [=] XGBOOST_DEVICE(size_t idx) { d_sorted_labels[idx] = d_labels[idx]; });
  dh::SegmentedSortKeys<true>(group_ptr, dh::ToSpan(sorted_labels));

  using IdxGroup = thrust::pair<size_t, size_t>;
  auto group_it = dh::MakeTransformIterator<IdxGroup>(
      thrust::make_counting_iterator(0ull), [=] XGBOOST_DEVICE(size_t idx) {
        return thrust::make_pair(idx, dh::SegmentId(group_ptr, idx));
      });
  auto value_it = dh::MakeTransformIterator<float>(
      group_it, [d_sorted_labels, group_ptr, truncation] XGBOOST_DEVICE(IdxGroup const &l) {
        auto g_begin = group_ptr[l.second];
        auto idx_in_group = l.first - g_begin;
        if (idx_in_group >= truncation) {
          // Truncating IDCG, this wastes some threads but IDCG is calcuated once per
          // dataset so should be fine.
          return 0.0f;
        }

        auto g_labels = d_sorted_labels.subspan(g_begin, group_ptr[l.second + 1] - g_begin);
        auto label = g_labels[idx_in_group];
        auto gain = ::xgboost::CalcNDCGGain(label);
        auto discount = CalcNDCGDiscount(idx_in_group);
        return gain * discount;
      });
  dh::XGBDeviceAllocator<common::byte> alloc;
  thrust::reduce_by_key(
      thrust::cuda::par(alloc), group_it, group_it + sorted_labels.size(), value_it,
      thrust::make_discard_iterator(), dh::tbegin(out_inv_IDCG),
      [] XGBOOST_DEVICE(IdxGroup const &l, IdxGroup const &r) { return l.second == r.second; },
      thrust::plus<float>{});
  dh::LaunchN(out_inv_IDCG.size(), [out_inv_IDCG] XGBOOST_DEVICE(size_t idx) {
    float idcg = out_inv_IDCG[idx];
    out_inv_IDCG[idx] = idcg == 0.0f ? 0.0f : 1.0f / idcg;
  });
}
}  // namespace obj
}  // namespace xgboost
#endif  // XGBOOST_OBJECTIVE_RANK_OBJ_CUH_
