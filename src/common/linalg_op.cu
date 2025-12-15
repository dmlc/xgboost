/**
 * Copyright 2025, XGBoost Contributors
 */
#include <thrust/scan.h>  // for inclusive_scan

#include <cstddef>  // for size_t

#include "algorithm.cuh"       // for ArgSort, RunLengthEncode
#include "device_helpers.cuh"  // for MakeIndexTransformIter
#include "device_vector.cuh"   // for DeviceUVector
#include "linalg_op.cuh"
#include "optional_weight.h"  // for OptionalWeights
#include "xgboost/linalg.h"   // for VectorView

namespace xgboost::linalg::cuda_impl {
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

template <typename T, std::int32_t D>
void Copy(Context const* ctx, TensorView<T const, D> in, linalg::Tensor<T, D>* p_out) {
  auto out = p_out->View(ctx->Device());
  if (in.CContiguous() && out.CContiguous()) {
    auto src = in.Values();
    auto dst = out.Values();
    // Use the size of the destination pointer as the input might be larger due to array slicing.
    dh::safe_cuda(cudaMemcpyAsync(dst.data(), src.data(), dst.size_bytes(), cudaMemcpyDefault,
                                  ctx->CUDACtx()->Stream()));
    return;
  }
  thrust::copy(ctx->CUDACtx()->CTP(), tcbegin(in), tcend(in), tbegin(out));
}
// explicit instantiations
template void Copy<float, 1>(Context const* ctx, TensorView<float const, 1> in,
                             linalg::Tensor<float, 1>* p_out);
template void Copy<float, 2>(Context const* ctx, TensorView<float const, 2> in,
                             linalg::Tensor<float, 2>* p_out);
}  // namespace xgboost::linalg::cuda_impl
