/*!
 * Copyright 2018 XGBoost contributors
 */
#include "common.h"

namespace xgboost {

int AllVisibleImpl::AllVisible() {
  int n_visgpus = 0;
  try {
    // When compiled with CUDA but running on CPU only device,
    // cudaGetDeviceCount will fail.
    dh::safe_cuda(cudaGetDeviceCount(&n_visgpus));
  } catch(const dmlc::Error &except) {
    return 0;
  }
  return n_visgpus;
}

int AllVisibleImpl::AllocateGPUDeviceId() {
  int device_ordinal;
  int n_devices;
  cudaError_t error = cudaFree(0);
  if (error != cudaSuccess) {
    dh::safe_cuda(cudaGetDeviceCount(&n_devices));
    // TODO - could add more checks here about usable devices
    // loop twice just in case hit race with someone else
    int n_iters = n_devices * 2;
    device_ordinal = 0;
    while (n_iters > 0) {
      dh::safe_cuda(cudaSetDevice(device_ordinal));
      // initialize a context
      error = cudaFree(0);
      if (error == cudaSuccess) {
        return device_ordinal;
      } else if (error == cudaErrorDevicesUnavailable) {
        // assume we lost the race or all are in use so try again
      } else {
        LOG(FATAL) << thrust::system_error(error, thrust::cuda_category(),
                                   std::string{__FILE__} + ": " +  // NOLINT
                                   std::to_string(__LINE__)).what();
      }
      n_iters--;
      if ((device_ordinal >= n_devices - 1) || (device_ordinal < 0)) {
        device_ordinal= 0;
      } else {
        device_ordinal++;
      }
    }
    LOG(FATAL) << "Error: could not allocate a GPU!" << std::endl;
    return 0;
  }
  dh::safe_cuda(cudaGetDevice(&device_ordinal));
  return device_ordinal;
}

}  // namespace xgboost
