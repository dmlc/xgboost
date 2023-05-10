/**
 * Copyright 2019-2023 by XGBoost Contributors
 * \file simple_dmatrix.cuh
 */
#ifndef XGBOOST_DATA_SIMPLE_DMATRIX_CUH_
#define XGBOOST_DATA_SIMPLE_DMATRIX_CUH_

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "../common/device_helpers.cuh"
#include "../common/error_msg.h"  // for InfInData
#include "device_adapter.cuh"     // for HasInfInData

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
void CopyDataToDMatrix(AdapterBatchT batch, common::Span<Entry> data,
                       float missing) {
  auto counting = thrust::make_counting_iterator(0llu);
  dh::XGBCachingDeviceAllocator<char> alloc;
  COOToEntryOp<decltype(batch)> transform_op{batch};
  thrust::transform_iterator<decltype(transform_op), decltype(counting)>
      transform_iter(counting, transform_op);
  auto begin_output = thrust::device_pointer_cast(data.data());
  dh::CopyIf(transform_iter, transform_iter + batch.Size(), begin_output,
             IsValidFunctor(missing));
}

template <typename AdapterBatchT>
void CountRowOffsets(const AdapterBatchT& batch, common::Span<bst_row_t> offset,
                     int device_idx, float missing) {
  dh::safe_cuda(cudaSetDevice(device_idx));
  IsValidFunctor is_valid(missing);
  // Count elements per row
  dh::LaunchN(batch.Size(), [=] __device__(size_t idx) {
    auto element = batch.GetElement(idx);
    if (is_valid(element)) {
      atomicAdd(reinterpret_cast<unsigned long long*>(  // NOLINT
                    &offset[element.row_idx]),
                static_cast<unsigned long long>(1));  // NOLINT
    }
  });

  dh::XGBCachingDeviceAllocator<char> alloc;
  thrust::exclusive_scan(thrust::cuda::par(alloc),
      thrust::device_pointer_cast(offset.data()),
      thrust::device_pointer_cast(offset.data() + offset.size()),
      thrust::device_pointer_cast(offset.data()));
}

template <typename AdapterBatchT>
size_t CopyToSparsePage(AdapterBatchT const& batch, int32_t device, float missing,
                        SparsePage* page) {
  bool valid = NoInfInData(batch, IsValidFunctor{missing});
  CHECK(valid) << error::InfInData();

  page->offset.SetDevice(device);
  page->data.SetDevice(device);
  page->offset.Resize(batch.NumRows() + 1);
  auto s_offset = page->offset.DeviceSpan();
  CountRowOffsets(batch, s_offset, device, missing);
  auto num_nonzero_ = page->offset.HostVector().back();
  page->data.Resize(num_nonzero_);
  CopyDataToDMatrix(batch, page->data.DeviceSpan(), missing);

  return num_nonzero_;
}
}  // namespace xgboost::data
#endif  // XGBOOST_DATA_SIMPLE_DMATRIX_CUH_
