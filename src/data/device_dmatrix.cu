/*!
 * Copyright 2020 by Contributors
 * \file device_dmatrix.cu
 * \brief Device-memory version of DMatrix.
 */

#include <xgboost/base.h>
#include <xgboost/data.h>

#include <memory>

#include "adapter.h"
#include "simple_dmatrix.h"
#include "device_dmatrix.h"
#include "device_adapter.cuh"
#include "../common/hist_util.h"
#include "../common/math.h"

namespace xgboost {
namespace data {

struct IsValidFunctor : public thrust::unary_function<Entry, bool> {
  explicit IsValidFunctor(float missing) : missing(missing) {}

  float missing;
  __device__ bool operator()(const data::COOTuple& e) const {
    if (common::CheckNAN(e.value) || e.value == missing) {
      return false;
    }
    return true;
  }
};

// Returns maximum row length
template <typename AdapterBatchT>
size_t CountRowOffsets(const AdapterBatchT& batch, common::Span<size_t> offset,
                     int device_idx, float missing) {
  // Count elements per row
  dh::LaunchN(device_idx, batch.Size(), [=] __device__(size_t idx) {
    auto element = batch.GetElement(idx);
    if (IsValid(element.value, missing)) {
      atomicAdd(reinterpret_cast<unsigned long long*>(  // NOLINT
                    &offset[element.row_idx]),
                static_cast<unsigned long long>(1));  // NOLINT
    }
  });
  size_t row_stride = thrust::reduce(dh::tbegin(offset), dh::tend(offset), thrust::maximum <size_t >());

  dh::XGBCachingDeviceAllocator<char> alloc;
  thrust::exclusive_scan(thrust::cuda::par(alloc),
      thrust::device_pointer_cast(offset.data()),
      thrust::device_pointer_cast(offset.data() + offset.size()),
      thrust::device_pointer_cast(offset.data()));
  return row_stride;
}

// Here the data is already correctly ordered and simply needs to be compacted
// to remove missing data
template <typename AdapterBatchT>
void CopyDataRowMajor(const AdapterBatchT& batch, EllpackPageImpl*dst,
                      int device_idx, float missing,
                      common::Span<size_t> row_ptr) {
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
DeviceDMatrix::DeviceDMatrix(AdapterT* adapter, float missing, int nthread) {
  common::HistogramCuts cuts = common::AdapterDeviceSketch(adapter, 256, missing);
  auto & batch = adapter->Value();
  // Work out how many valid entries we have in each row
  dh::caching_device_vector<size_t> row_ptr(adapter->NumRows() + 1,
                                                      0);
  common::Span<size_t > row_ptr_span( row_ptr.data().get(),row_ptr.size() );
  size_t row_stride=CountRowOffsets(batch, row_ptr_span, adapter->DeviceIdx(), missing);

  info.num_nonzero_ = row_ptr.back();// Device to host copy
  info.num_col_ = adapter->NumColumns();
  info.num_row_ = adapter->NumRows();
  ellpack_page_.reset(new EllpackPage());
  auto impl = ellpack_page_->Impl(adapter->DeviceIdx(),cuts,this->IsDense(),row_stride, adapter->NumRows());
  if (adapter->IsRowMajor()) {
    CopyDataRowMajor(batch, impl, adapter->DeviceIdx(), row_ptr_span);
  }

  //*impl = EllpackPageImpl(adapter->DeviceIdx(),cuts ,info.num_nonzero_==info.num_col_*info.num_row_,,info.num_row_);
  // Synchronise worker columns
  rabit::Allreduce<rabit::op::Max>(&info.num_col_, 1);
}
template DeviceDMatrix::DeviceDMatrix(CudfAdapter* adapter, float missing,
                                      int nthread);
template DeviceDMatrix::DeviceDMatrix(CupyAdapter* adapter, float missing,
                                      int nthread);
}  // namespace data
}  // namespace xgboost
