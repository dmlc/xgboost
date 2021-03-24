/*!
 * Copyright 2021 by Contributors
 */
#include "array_interface.h"
#include "../common/common.h"

namespace xgboost {
void ArrayInterfaceHandler::SyncCudaStream(int64_t stream) {
  switch (stream) {
  case 0:
    LOG(FATAL) << "Invalid stream ID in array interface: " << stream;
  case 1:
    // default legacy stream
    break;
  case 2:
    // default per-thread stream
  default:
    dh::safe_cuda(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
  }
}
}  // namespace xgboost
