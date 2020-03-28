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

template <typename AdapterT>
void CopyDataColumnMajor(AdapterT* adapter, common::Span<Entry> data,
                         int device_idx, float missing,
                         common::Span<size_t> row_ptr) {
  // Step 1: Get the sizes of the input columns
  dh::device_vector<size_t> column_sizes(adapter->NumColumns());
  auto d_column_sizes = column_sizes.data().get();
  auto& batch = adapter->Value();
  // Populate column sizes
  dh::LaunchN(device_idx, batch.Size(), [=] __device__(size_t idx) {
    const auto& e = batch.GetElement(idx);
    atomicAdd(reinterpret_cast<unsigned long long*>(  // NOLINT
                  &d_column_sizes[e.column_idx]),
              static_cast<unsigned long long>(1));  // NOLINT
  });

  thrust::host_vector<size_t> host_column_sizes = column_sizes;

  // Step 2: Iterate over columns, place elements in correct row, increment
  // temporary row pointers
  dh::device_vector<size_t> temp_row_ptr(
      thrust::device_pointer_cast(row_ptr.data()),
      thrust::device_pointer_cast(row_ptr.data() + row_ptr.size()));
  auto d_temp_row_ptr = temp_row_ptr.data().get();
  size_t begin = 0;
  IsValidFunctor is_valid(missing);
  for (auto size : host_column_sizes) {
    size_t end = begin + size;
    dh::LaunchN(device_idx, end - begin, [=] __device__(size_t idx) {
      const auto& e = batch.GetElement(idx + begin);
      if (!is_valid(e)) return;
      data[d_temp_row_ptr[e.row_idx]] = Entry(e.column_idx, e.value);
      d_temp_row_ptr[e.row_idx] += 1;
    });

    begin = end;
  }
}

// Here the data is already correctly ordered and simply needs to be compacted
// to remove missing data
template <typename AdapterT>
void CopyDataRowMajor(AdapterT* adapter, common::Span<Entry> data,
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
  info.num_nonzero_ = sparse_page_.offset.HostVector().back();
  sparse_page_.data.Resize(info.num_nonzero_);
  if (adapter->IsRowMajor()) {
    CopyDataRowMajor(adapter, sparse_page_.data.DeviceSpan(),
                        adapter->DeviceIdx(), missing, s_offset);
  } else {
    CopyDataColumnMajor(adapter, sparse_page_.data.DeviceSpan(),
                        adapter->DeviceIdx(), missing, s_offset);
  }
  // Sync
  sparse_page_.data.HostVector();

  info.num_col_ = adapter->NumColumns();
  info.num_row_ = adapter->NumRows();
  // Synchronise worker columns
  rabit::Allreduce<rabit::op::Max>(&info.num_col_, 1);
}

template SimpleDMatrix::SimpleDMatrix(CudfAdapter* adapter, float missing,
                                      int nthread);
template SimpleDMatrix::SimpleDMatrix(CupyAdapter* adapter, float missing,
                                      int nthread);
}  // namespace data
}  // namespace xgboost
