/**
 * Copyright 2021-2023, XGBoost contributors
 */
#include "../common/device_helpers.cuh"  // for CurrentDevice
#include "proxy_dmatrix.cuh"             // for Dispatch, DMatrixProxy
#include "simple_dmatrix.cuh"            // for CopyToSparsePage
#include "sparse_page_source.h"
#include "xgboost/data.h"  // for SparsePage

namespace xgboost::data {
namespace detail {
std::size_t NSamplesDevice(DMatrixProxy *proxy) {
  return cuda_impl::Dispatch(proxy, [](auto const &value) { return value.NumRows(); });
}

std::size_t NFeaturesDevice(DMatrixProxy *proxy) {
  return cuda_impl::Dispatch(proxy, [](auto const &value) { return value.NumCols(); });
}
}  // namespace detail

void DevicePush(DMatrixProxy *proxy, float missing, SparsePage *page) {
  auto device = proxy->Device();
  if (!device.IsCUDA()) {
    device = DeviceOrd::CUDA(dh::CurrentDevice());
  }
  CHECK(device.IsCUDA());

  cuda_impl::Dispatch(proxy,
                      [&](auto const &value) { CopyToSparsePage(value, device, missing, page); });
}
}  // namespace xgboost::data
