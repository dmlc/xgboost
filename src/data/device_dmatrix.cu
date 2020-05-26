/*!
 * Copyright 2020 by Contributors
 * \file device_dmatrix.cu
 * \brief Device-memory version of DMatrix.
 */

#include <thrust/execution_policy.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <xgboost/base.h>
#include <xgboost/data.h>
#include <memory>
#include <utility>
#include "../common/hist_util.h"
#include "adapter.h"
#include "device_adapter.cuh"
#include "ellpack_page.cuh"
#include "device_dmatrix.h"

namespace xgboost {
namespace data {
// Does not currently support metainfo as no on-device data source contains this
// Current implementation assumes a single batch. More batches can
// be supported in future. Does not currently support inferring row/column size
template <typename AdapterT>
DeviceDMatrix::DeviceDMatrix(AdapterT* adapter, float missing, int nthread, int max_bin) {
  dh::safe_cuda(cudaSetDevice(adapter->DeviceIdx()));
  auto& batch = adapter->Value();
  // Work out how many valid entries we have in each row
  dh::caching_device_vector<size_t> row_counts(adapter->NumRows() + 1, 0);
  common::Span<size_t> row_counts_span(row_counts.data().get(),
                                       row_counts.size());
  size_t row_stride =
      GetRowCounts(batch, row_counts_span, adapter->DeviceIdx(), missing);

  dh::XGBCachingDeviceAllocator<char> alloc;
  info_.num_nonzero_ = thrust::reduce(thrust::cuda::par(alloc),
                                      row_counts.begin(), row_counts.end());
  info_.num_col_ = adapter->NumColumns();
  info_.num_row_ = adapter->NumRows();

  ellpack_page_.reset(new EllpackPage());
  *ellpack_page_->Impl() =
      EllpackPageImpl(adapter, missing, this->IsDense(), nthread, max_bin,
                      row_counts_span, row_stride);

  // Synchronise worker columns
  rabit::Allreduce<rabit::op::Max>(&info_.num_col_, 1);
}

#define DEVICE_DMARIX_SPECIALIZATION(__ADAPTER_T)                       \
  template DeviceDMatrix::DeviceDMatrix(__ADAPTER_T* adapter, float missing, \
                                        int nthread, int max_bin);

DEVICE_DMARIX_SPECIALIZATION(CudfAdapter);
DEVICE_DMARIX_SPECIALIZATION(CupyAdapter);
}  // namespace data
}  // namespace xgboost
