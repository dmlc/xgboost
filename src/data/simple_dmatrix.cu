/*!
 * Copyright 2019-2021 by XGBoost Contributors
 * \file simple_dmatrix.cu
 */
#include <thrust/copy.h>
#include <xgboost/data.h>
#include "simple_dmatrix.cuh"
#include "simple_dmatrix.h"
#include "device_adapter.cuh"

namespace xgboost {
namespace data {

// Does not currently support metainfo as no on-device data source contains this
// Current implementation assumes a single batch. More batches can
// be supported in future. Does not currently support inferring row/column size
template <typename AdapterT>
SimpleDMatrix::SimpleDMatrix(AdapterT* adapter, float missing, int32_t /*nthread*/) {
  auto device = (adapter->DeviceIdx() < 0 || adapter->NumRows() == 0) ? dh::CurrentDevice()
                                                                      : adapter->DeviceIdx();
  CHECK_GE(device, 0);
  dh::safe_cuda(cudaSetDevice(device));

  CHECK(adapter->NumRows() != kAdapterUnknownSize);
  CHECK(adapter->NumColumns() != kAdapterUnknownSize);

  adapter->BeforeFirst();
  adapter->Next();

  // Enforce single batch
  CHECK(!adapter->Next());

  info_.num_nonzero_ =
      CopyToSparsePage(adapter->Value(), device, missing, sparse_page_.get());
  info_.num_col_ = adapter->NumColumns();
  info_.num_row_ = adapter->NumRows();
  // Synchronise worker columns
  collective::Allreduce<collective::Operation::kMax>(&info_.num_col_, 1);
}

template SimpleDMatrix::SimpleDMatrix(CudfAdapter* adapter, float missing,
                                      int nthread);
template SimpleDMatrix::SimpleDMatrix(CupyAdapter* adapter, float missing,
                                      int nthread);
}  // namespace data
}  // namespace xgboost
