/*!
 * Copyright 2019 by Contributors
 * \file simple_dmatrix.cu
 */
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <xgboost/data.h>
#include "../common/random.h"
#include "./simple_dmatrix.h"
#include "device_adapter.cuh"

namespace xgboost {
namespace data {


template <typename AdapterBatchT>
void CountRowOffsets(const AdapterBatchT& batch, common::Span<bst_row_t> offset,
                     int device_idx, float missing) {
  IsValidFunctor is_valid(missing);
  // Count elements per row
  dh::LaunchN(device_idx, batch.Size(), [=] __device__(size_t idx) {
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

// Here the data is already correctly ordered and simply needs to be compacted
// to remove missing data
template <typename AdapterT>
void CopyDataToDMatrix(AdapterT* adapter, common::Span<Entry> data,
                       int device_idx, float missing,
                       common::Span<size_t> row_ptr) {
  auto& batch = adapter->Value();
  auto transform_f = [=] __device__(size_t idx) {
    const auto& e = batch.GetElement(idx);
    return Entry(e.column_idx, e.value);
  };  // NOLINT
  auto counting = thrust::make_counting_iterator(0llu);
  thrust::transform_iterator<decltype(transform_f), decltype(counting), Entry>
      transform_iter(counting, transform_f);
  dh::XGBCachingDeviceAllocator<char> alloc;
  thrust::copy_if(
      thrust::cuda::par(alloc), transform_iter, transform_iter + batch.Size(),
      thrust::device_pointer_cast(data.data()), IsValidFunctor(missing));
}

// Does not currently support metainfo as no on-device data source contains this
// Current implementation assumes a single batch. More batches can
// be supported in future. Does not currently support inferring row/column size
template <typename AdapterT>
SimpleDMatrix::SimpleDMatrix(AdapterT* adapter, float missing, int nthread) {
  dh::safe_cuda(cudaSetDevice(adapter->DeviceIdx()));
  CHECK(adapter->NumRows() != kAdapterUnknownSize);
  CHECK(adapter->NumColumns() != kAdapterUnknownSize);

  adapter->BeforeFirst();
  adapter->Next();
  auto& batch = adapter->Value();
  sparse_page_.offset.SetDevice(adapter->DeviceIdx());
  sparse_page_.data.SetDevice(adapter->DeviceIdx());

  // Enforce single batch
  CHECK(!adapter->Next());
  sparse_page_.offset.Resize(adapter->NumRows() + 1);
  auto s_offset = sparse_page_.offset.DeviceSpan();
  CountRowOffsets(batch, s_offset, adapter->DeviceIdx(), missing);
  info_.num_nonzero_ = sparse_page_.offset.HostVector().back();
  sparse_page_.data.Resize(info_.num_nonzero_);
  CopyDataToDMatrix(adapter, sparse_page_.data.DeviceSpan(),
                    adapter->DeviceIdx(), missing, s_offset);

  info_.num_col_ = adapter->NumColumns();
  info_.num_row_ = adapter->NumRows();
  // Synchronise worker columns
  rabit::Allreduce<rabit::op::Max>(&info_.num_col_, 1);
}

template SimpleDMatrix::SimpleDMatrix(CudfAdapter* adapter, float missing,
                                      int nthread);
template SimpleDMatrix::SimpleDMatrix(CupyAdapter* adapter, float missing,
                                      int nthread);
}  // namespace data
}  // namespace xgboost
