/**
 * Copyright 2019-2023, XGBoost Contributors
 * \file simple_dmatrix.cu
 */
#include <thrust/copy.h>

#include "device_adapter.cuh"  // for CurrentDevice
#include "simple_dmatrix.cuh"
#include "simple_dmatrix.h"
#include "xgboost/context.h"  // for Context
#include "xgboost/data.h"

namespace xgboost {
namespace data {

// Does not currently support metainfo as no on-device data source contains this
// Current implementation assumes a single batch. More batches can
// be supported in future. Does not currently support inferring row/column size
template <typename AdapterT>
SimpleDMatrix::SimpleDMatrix(AdapterT* adapter, float missing, std::int32_t nthread,
                             DataSplitMode data_split_mode) {
  CHECK(data_split_mode != DataSplitMode::kCol)
      << "Column-wise data split is currently not supported on the GPU.";
  auto device = (adapter->DeviceIdx() < 0 || adapter->NumRows() == 0) ? dh::CurrentDevice()
                                                                      : adapter->DeviceIdx();
  CHECK_GE(device, 0);
  dh::safe_cuda(cudaSetDevice(device));

  Context ctx;
  ctx.Init(Args{{"nthread", std::to_string(nthread)}, {"gpu_id", std::to_string(device)}});

  CHECK(adapter->NumRows() != kAdapterUnknownSize);
  CHECK(adapter->NumColumns() != kAdapterUnknownSize);

  adapter->BeforeFirst();
  adapter->Next();

  // Enforce single batch
  CHECK(!adapter->Next());

  info_.num_nonzero_ = CopyToSparsePage(adapter->Value(), device, missing, sparse_page_.get());
  info_.num_col_ = adapter->NumColumns();
  info_.num_row_ = adapter->NumRows();
  // Synchronise worker columns
  info_.data_split_mode = data_split_mode;
  info_.SynchronizeNumberOfColumns();

  this->fmat_ctx_ = ctx;
}

template SimpleDMatrix::SimpleDMatrix(CudfAdapter* adapter, float missing,
                                      int nthread, DataSplitMode data_split_mode);
template SimpleDMatrix::SimpleDMatrix(CupyAdapter* adapter, float missing,
                                      int nthread, DataSplitMode data_split_mode);
}  // namespace data
}  // namespace xgboost
