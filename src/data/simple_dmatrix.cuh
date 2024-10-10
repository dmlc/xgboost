/**
 * Copyright 2019-2024, XGBoost Contributors
 * \file simple_dmatrix.cuh
 */
#ifndef XGBOOST_DATA_SIMPLE_DMATRIX_CUH_
#define XGBOOST_DATA_SIMPLE_DMATRIX_CUH_

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "../common/device_helpers.cuh"
#include "../common/error_msg.h"  // for InfInData
#include "../common/algorithm.cuh"  // for CopyIf
#include "device_adapter.cuh"     // for NoInfInData

namespace xgboost::data {

template <typename AdapterBatchT>
struct COOToEntryOp {
  AdapterBatchT batch;
  __device__ Entry operator()(size_t idx) {
    const auto& e = batch.GetElement(idx);
    return Entry(e.column_idx, e.value);
  }
};

// Here the data is already correctly ordered and simply needs to be compacted
// to remove missing data
template <typename AdapterBatchT>
void CopyDataToDMatrix(Context const* ctx, AdapterBatchT batch, common::Span<Entry> data,
                       float missing) {
  auto counting = thrust::make_counting_iterator(0llu);
  COOToEntryOp<decltype(batch)> transform_op{batch};
  thrust::transform_iterator<decltype(transform_op), decltype(counting)> transform_iter(
      counting, transform_op);
  auto begin_output = thrust::device_pointer_cast(data.data());
  common::CopyIf(ctx->CUDACtx(), transform_iter, transform_iter + batch.Size(), begin_output,
                 IsValidFunctor(missing));
}

template <typename AdapterBatchT>
void CountRowOffsets(Context const* ctx, const AdapterBatchT& batch, common::Span<bst_idx_t> offset,
                     DeviceOrd device, float missing) {
  dh::safe_cuda(cudaSetDevice(device.ordinal));
  IsValidFunctor is_valid(missing);
  auto cuctx = ctx->CUDACtx();
  // Count elements per row
  dh::LaunchN(batch.Size(), cuctx->Stream(), [=] __device__(size_t idx) {
    auto element = batch.GetElement(idx);
    if (is_valid(element)) {
      atomicAdd(reinterpret_cast<unsigned long long*>(  // NOLINT
                    &offset[element.row_idx]),
                static_cast<unsigned long long>(1));  // NOLINT
    }
  });

  thrust::exclusive_scan(cuctx->CTP(), thrust::device_pointer_cast(offset.data()),
                         thrust::device_pointer_cast(offset.data() + offset.size()),
                         thrust::device_pointer_cast(offset.data()));
}

template <typename AdapterBatchT>
bst_idx_t CopyToSparsePage(Context const* ctx, AdapterBatchT const& batch, DeviceOrd device,
                           float missing, SparsePage* page) {
  bool valid = NoInfInData(ctx, batch, IsValidFunctor{missing});
  CHECK(valid) << error::InfInData();

  page->offset.SetDevice(device);
  page->data.SetDevice(device);
  page->offset.Resize(batch.NumRows() + 1);
  auto s_offset = page->offset.DeviceSpan();
  CountRowOffsets(ctx, batch, s_offset, device, missing);
  auto num_nonzero_ = page->offset.HostVector().back();
  page->data.Resize(num_nonzero_);
  CopyDataToDMatrix(ctx, batch, page->data.DeviceSpan(), missing);

  return num_nonzero_;
}
}  // namespace xgboost::data
#endif  // XGBOOST_DATA_SIMPLE_DMATRIX_CUH_
