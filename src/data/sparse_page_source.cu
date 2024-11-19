/**
 * Copyright 2021-2024, XGBoost contributors
 */
#include "../common/device_helpers.cuh"  // for CurrentDevice
#include "proxy_dmatrix.cuh"             // for Dispatch, DMatrixProxy
#include "simple_dmatrix.cuh"            // for CopyToSparsePage
#include "sparse_page_source.h"
#include "xgboost/data.h"  // for SparsePage

namespace xgboost::data {
void DevicePush(DMatrixProxy *proxy, float missing, SparsePage *page) {
  auto device = proxy->Device();
  if (!device.IsCUDA()) {
    device = DeviceOrd::CUDA(dh::CurrentDevice());
  }
  CHECK(device.IsCUDA());
  auto ctx = Context{}.MakeCUDA(device.ordinal);

  cuda_impl::Dispatch(
      proxy, [&](auto const &value) { CopyToSparsePage(&ctx, value, device, missing, page); });
}
}  // namespace xgboost::data
