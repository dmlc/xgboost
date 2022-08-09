/*!
 * Copyright 2021 by Contributors
 * \file proxy_dmatrix.cc
 */

#include "proxy_dmatrix.h"

namespace xgboost {
namespace data {
void DMatrixProxy::SetCUDAArray(char const *c_interface) {
  common::AssertGPUSupport();
#if defined(XGBOOST_USE_CUDA)
  StringView interface_str{c_interface};
  Json json_array_interface = Json::Load(interface_str);
  if (IsA<Array>(json_array_interface)) {
    this->FromCudaColumnar(interface_str);
  } else {
    this->FromCudaArray(interface_str);
  }
#else
  (void*)c_interface;
#endif  // defined(XGBOOST_USE_CUDA)
}

void DMatrixProxy::SetArrayData(char const *c_interface) {
  std::shared_ptr<ArrayAdapter> adapter{new ArrayAdapter(StringView{c_interface})};
  this->batch_ = adapter;
  this->Info().num_col_ = adapter->NumColumns();
  this->Info().num_row_ = adapter->NumRows();
  this->ctx_.gpu_id = Context::kCpuId;
}

void DMatrixProxy::SetCSRData(char const *c_indptr, char const *c_indices,
                              char const *c_values, bst_feature_t n_features, bool on_host) {
  CHECK(on_host) << "Not implemented on device.";
  std::shared_ptr<CSRArrayAdapter> adapter{new CSRArrayAdapter(
      StringView{c_indptr}, StringView{c_indices}, StringView{c_values}, n_features)};
  this->batch_ = adapter;
  this->Info().num_col_ = adapter->NumColumns();
  this->Info().num_row_ = adapter->NumRows();
  this->ctx_.gpu_id = Context::kCpuId;
}
}  // namespace data
}  // namespace xgboost
