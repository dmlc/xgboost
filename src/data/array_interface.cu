/*!
 * Copyright 2021 by Contributors
 */
#include "array_interface.h"
#include "../common/common.h"

namespace xgboost {
void ArrayInterfaceHandler::SyncCudaStream(ptrdiff_t stream) {
  dh::safe_cuda(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
}
}  // namespace xgboost
