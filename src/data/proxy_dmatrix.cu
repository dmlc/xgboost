/*!
 * Copyright 2020 XGBoost contributors
 */
#include "proxy_dmatrix.h"
#include "device_adapter.cuh"

namespace xgboost {
namespace data {

void DMatrixProxy::FromCudaColumnar(std::string interface_str) {
  std::shared_ptr<data::CudfAdapter> adapter {new data::CudfAdapter(interface_str)};
  auto const& value = adapter->Value();
  this->batch_ = adapter;
  device_ = adapter->DeviceIdx();
}

void DMatrixProxy::FromCudaArray(std::string interface_str) {
  std::shared_ptr<CupyAdapter> adapter(new CupyAdapter(interface_str));
  this->batch_ = adapter;
  device_ = adapter->DeviceIdx();
}

}  // namespace data
}  // namespace xgboost
