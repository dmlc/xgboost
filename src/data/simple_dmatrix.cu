/*!
 * Copyright 2019-2021 by XGBoost Contributors
 * \file simple_dmatrix.cu
 */
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <xgboost/data.h>
#include "../common/random.h"
#include "simple_dmatrix.cuh"
#include "./simple_dmatrix.h"
#include "device_adapter.cuh"

namespace xgboost {
namespace data {

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

  // Enforce single batch
  CHECK(!adapter->Next());

  info_.num_nonzero_ = CopyToSparsePage(adapter->Value(), adapter->DeviceIdx(),
                                        missing, &sparse_page_);
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
