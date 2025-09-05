/**
 * Copyright 2025, XGBoost Contributors
 */
#include <thrust/for_each.h>                    // for for_each_n
#include <thrust/iterator/counting_iterator.h>  // for make_counting_iterator
#include <thrust/scan.h>                        // for inclusive_scan

#include <cstddef>  // for size_t

#include "algorithm.cuh"       // for ArgSort, RunLengthEncode
#include "device_helpers.cuh"  // for MakeIndexTransformIter
#include "device_vector.cuh"   // for DeviceUVector
#include "linalg_op.cuh"
#include "optional_weight.h"  // for OptionalWeights
#include "xgboost/linalg.h"   // for VectorView

namespace xgboost::linalg::cuda_impl {
void VecScaMul(Context const* ctx, linalg::VectorView<float> x, double mul) {
  thrust::for_each_n(ctx->CUDACtx()->CTP(), thrust::make_counting_iterator(0ul), x.Size(),
                     [=] XGBOOST_DEVICE(std::size_t i) mutable { x(i) = x(i) * mul; });
}

void SmallHistogram(Context const* ctx, linalg::MatrixView<float const> indices,
                    common::OptionalWeights const& d_weights, linalg::VectorView<float> bins) {
  auto n_bins = bins.Size();
  auto cuctx = ctx->CUDACtx();
  // Sort for segmented sum
  dh::DeviceUVector<std::size_t> sorted_idx(indices.Size());
  common::ArgSort<true>(ctx, indices.Values(), dh::ToSpan(sorted_idx));
  auto d_sorted_idx = dh::ToSpan(sorted_idx);

  auto key_it = dh::MakeIndexTransformIter(
      [=] XGBOOST_DEVICE(std::size_t i) { return indices(d_sorted_idx[i]); });

  dh::device_vector<std::size_t> counts_out(n_bins + 1, 0);
  // Obtain the segment boundaries for the segmented sum.
  dh::DeviceUVector<float> unique(n_bins);
  dh::CachingDeviceUVector<std::size_t> num_runs(1);
  common::RunLengthEncode(cuctx->Stream(), key_it, unique.begin(), counts_out.begin() + 1,
                          num_runs.begin(), indices.Size());
  thrust::inclusive_scan(cuctx->CTP(), counts_out.begin(), counts_out.end(), counts_out.begin());

  auto val_it = dh::MakeIndexTransformIter(
      [=] XGBOOST_DEVICE(std::size_t i) { return d_weights[d_sorted_idx[i]]; });
  // Sum weighted-label for each class to acc, counts_out is the segment ptr after inclusive_scan
  common::SegmentedSum(cuctx->Stream(), val_it, linalg::tbegin(bins), n_bins, counts_out.cbegin(),
                       counts_out.cbegin() + 1);
}
}  // namespace xgboost::linalg::cuda_impl
