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
template <typename AdapterT>
void CopyDataToDMatrix(AdapterT* adapter, common::Span<Entry> data,
                       float missing) {
  auto batch = adapter->Value();
  auto counting = thrust::make_counting_iterator(0llu);
  dh::XGBCachingDeviceAllocator<char> alloc;
  COOToEntryOp<decltype(batch)> transform_op{batch};
  thrust::transform_iterator<decltype(transform_op), decltype(counting)>
      transform_iter(counting, transform_op);
  // We loop over batches because thrust::copy_if cant deal with sizes > 2^31
  // See thrust issue #1302
  size_t max_copy_size = std::numeric_limits<int>::max() / 2;
  auto begin_output = thrust::device_pointer_cast(data.data());
  for (size_t offset = 0; offset < batch.Size(); offset += max_copy_size) {
    auto begin_input = transform_iter + offset;
    auto end_input =
        transform_iter + std::min(offset + max_copy_size, batch.Size());
    begin_output =
        thrust::copy_if(thrust::cuda::par(alloc), begin_input, end_input,
                        begin_output, IsValidFunctor(missing));
  }
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
  CopyDataToDMatrix(adapter, sparse_page_.data.DeviceSpan(), missing);

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
