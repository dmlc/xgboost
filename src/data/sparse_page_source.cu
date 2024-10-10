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

void InitNewThread::operator()() const {
  *GlobalConfigThreadLocalStore::Get() = config;
  // For CUDA 12.2, we need to force initialize the CUDA context by synchronizing the
  // stream when creating a new thread in the thread pool. While for CUDA 11.8, this
  // action might cause an insufficient driver version error for some reason. Lastly, it
  // should work with CUDA 12.5 without any action being taken.

  // dh::DefaultStream().Sync();
}
}  // namespace xgboost::data
