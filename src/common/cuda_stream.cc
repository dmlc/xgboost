/**
 * Copyright 2024-2026, XGBoost contributors
 */
#if defined(XGBOOST_USE_CUDA)
#include "cuda_stream.h"

#include "cuda_rt_utils.h"  // for CurrentDevice
#include "utils.h"

namespace xgboost::curt {
Stream::Stream(std::int32_t device) {
  std::int32_t cur = CurrentDevice();
  auto guard = common::MakeCleanup([=] { SetDevice(cur); });
  SetDevice(device);
  dh::safe_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
}
}  // namespace xgboost::curt
#endif  // defined(XGBOOST_USE_CUDA)
