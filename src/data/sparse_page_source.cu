/*!
 * Copyright 2021 XGBoost contributors
 */
#include "sparse_page_source.h"
#include "proxy_dmatrix.cuh"
#include "simple_dmatrix.cuh"

namespace xgboost {
namespace data {

namespace detail {
size_t NSamplesDevice(DMatrixProxy *proxy) {
  return Dispatch(proxy, [](auto const &value) { return value.NumRows(); });
}

size_t NFeaturesDevice(DMatrixProxy *proxy) {
  return Dispatch(proxy, [](auto const &value) { return value.NumCols(); });
}
}  // namespace detail

void DevicePush(DMatrixProxy* proxy, float missing, SparsePage* page) {
  auto device = proxy->DeviceIdx();
  Dispatch(proxy, [&](auto const &value) {
    CopyToSparsePage(value, device, missing, page);
  });
}
}  // namespace data
}  // namespace xgboost
