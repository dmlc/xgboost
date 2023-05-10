/**
 * Copyright 2021-2023, XGBoost Contributors
 */
#include <cstdint>  // for int64_t

#include "../common/common.h"
#include "../common/device_helpers.cuh"  // for DefaultStream, CUDAEvent
#include "array_interface.h"
#include "xgboost/logging.h"

namespace xgboost {
void ArrayInterfaceHandler::SyncCudaStream(std::int64_t stream) {
  switch (stream) {
    case 0:
      /**
       * disallowed by the `__cuda_array_interface__`.  Quote:
       *
       *   This is disallowed as it would be ambiguous between None and the default
       *   stream, and also between the legacy and per-thread default streams. Any use
       *   case where 0 might be given should either use None, 1, or 2 instead for
       *   clarity.
       */
      LOG(FATAL) << "Invalid stream ID in array interface: " << stream;
    case 1:
      // default legacy stream
      break;
    case 2:
      // default per-thread stream
    default: {
      dh::CUDAEvent e;
      e.Record(dh::CUDAStreamView{reinterpret_cast<cudaStream_t>(stream)});
      dh::DefaultStream().Wait(e);
    }
  }
}

bool ArrayInterfaceHandler::IsCudaPtr(void const* ptr) {
  if (!ptr) {
    return false;
  }
  cudaPointerAttributes attr;
  auto err = cudaPointerGetAttributes(&attr, ptr);
  // reset error
  CHECK_EQ(err, cudaGetLastError());
  if (err == cudaErrorInvalidValue) {
    // CUDA < 11
    return false;
  } else if (err == cudaSuccess) {
    // CUDA >= 11
    switch (attr.type) {
      case cudaMemoryTypeUnregistered:
      case cudaMemoryTypeHost:
        return false;
      default:
        return true;
    }
    return true;
  } else {
    // other errors, `cudaErrorNoDevice`, `cudaErrorInsufficientDriver` etc.
    return false;
  }
}
}  // namespace xgboost
