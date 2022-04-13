#ifndef XGBOOST_COMMON_STATS_CUH_
#define XGBOOST_COMMON_STATS_CUH_

#include <thrust/sort.h>
#include "device_helpers.cuh"
#include "linalg_op.cuh"
#include "xgboost/generic_parameters.h"
#include "xgboost/tree_model.h"

namespace xgboost {
namespace common {
void SegmentedPercentile(Context const* ctx, double alpha, RowIndexCache const& row_index,
                         MetaInfo const& info, HostDeviceVector<float> const& predt,
                         HostDeviceVector<float>* quantiles) {
  CHECK(alpha >= 0 && alpha <= 1);

  auto d_predt = predt.ConstDeviceSpan();
  auto d_labels = info.labels.View(ctx->gpu_id);
  linalg::Tensor<float, 2> residue{d_labels.Shape(), ctx->gpu_id};
  auto d_residue = residue.View(ctx->gpu_id);
  CHECK_EQ(d_predt.size(), d_labels.Size());
  linalg::ElementWiseKernel(ctx, d_labels, [=] XGBOOST_DEVICE(size_t i, float y) mutable {
    auto idx = linalg::UnravelIndex(i, d_labels.Shape());
    size_t sample_id = std::get<0>(idx);
    size_t target_id = std::get<1>(idx);
    d_residue(sample_id, target_id) = y - d_predt[i];
  });

  dh::device_vector<size_t> sorted_idx(d_labels.Shape(0));
  dh::Iota(dh::ToSpan(sorted_idx));

  using Tup = thrust::tuple<size_t, float>;
  auto d_leaf_ptr = row_index.node_ptr.ConstDeviceSpan();
  auto d_row_index = row_index.row_index.ConstDeviceSpan();
  auto key_it = dh::MakeTransformIterator<Tup>(
      thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(size_t i) -> Tup {
        auto idx = linalg::UnravelIndex(i, d_labels.Shape());
        size_t sample_id = std::get<0>(idx);
        size_t target_id = std::get<1>(idx);
        auto leaf_idx = dh::SegmentId(d_leaf_ptr, sample_id);
        auto residue = d_residue(d_row_index[sample_id], target_id);
        return thrust::make_tuple(leaf_idx, residue);
      });
  dh::device_vector<Tup> keys(residue.Size());
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

  quantiles->SetDevice(ctx->gpu_id);
  quantiles->Resize(row_index.node_idx.Size());
  auto d_results = quantiles->DeviceSpan();

  auto d_sorted_idx = dh::ToSpan(sorted_idx);
  auto d_keys = dh::ToSpan(keys);

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
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_STATS_CUH_
