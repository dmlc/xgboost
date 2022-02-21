/*!
 * Copyright 2020-2022, XGBoost contributors
 */
#include "proxy_dmatrix.h"
#include "device_adapter.cuh"

namespace xgboost {
namespace data {

void DMatrixProxy::FromCudaColumnar(std::string interface_str) {
  std::shared_ptr<data::CudfAdapter> adapter {new data::CudfAdapter(interface_str)};
  auto const& value = adapter->Value();
  this->batch_ = adapter;
  ctx_.gpu_id = adapter->DeviceIdx();
  this->Info().num_col_ = adapter->NumColumns();
  this->Info().num_row_ = adapter->NumRows();
  if (ctx_.gpu_id < 0) {
    CHECK_EQ(this->Info().num_row_, 0);
  }
}

void DMatrixProxy::FromCudaArray(std::string interface_str) {
  std::shared_ptr<CupyAdapter> adapter(new CupyAdapter(interface_str));
  this->batch_ = adapter;
  ctx_.gpu_id = adapter->DeviceIdx();
  this->Info().num_col_ = adapter->NumColumns();
  this->Info().num_row_ = adapter->NumRows();
  if (ctx_.gpu_id < 0) {
    CHECK_EQ(this->Info().num_row_, 0);
  }
}
}  // namespace data
}  // namespace xgboost
