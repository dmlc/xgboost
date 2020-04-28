/*!
 * Copyright 2020 XGBoost contributors
 */
#include "proxy_dmatrix.h"
#include "device_adapter.cuh"

namespace xgboost {
namespace data {

void DMatrixProxy::FromCudaColumnar(std::string interface_str) {
  data::CudfAdapter adapter(interface_str);
  this->batch_ = adapter;
  device_ = adapter.DeviceIdx();
}

void DMatrixProxy::FromCudaArray(std::string interface_str) {
  CupyAdapter adapter(interface_str);
  this->batch_ = adapter;
  CHECK(this->batch_.type() == typeid(CupyAdapter));
  device_ = adapter.DeviceIdx();
}

}  // namespace data
}  // namespace xgboost
