/**
 * Copyright 2015-2024, XGBoost Contributors
 */
#include "cuda_rt_utils.h"

#if defined(XGBOOST_USE_CUDA)
#include <cuda_runtime_api.h>
#endif  // defined(XGBOOST_USE_CUDA)

#include <cstdint>  // for int32_t

#include "common.h"  // for safe_cuda

namespace xgboost::common {
#if defined(XGBOOST_USE_CUDA)
std::int32_t AllVisibleGPUs() {
  int n_visgpus = 0;
  try {
    // When compiled with CUDA but running on CPU only device,
    // cudaGetDeviceCount will fail.
    dh::safe_cuda(cudaGetDeviceCount(&n_visgpus));
  } catch (const dmlc::Error &) {
    cudaGetLastError();  // reset error.
    return 0;
  }
  return n_visgpus;
}

std::int32_t CurrentDevice() {
  std::int32_t device = 0;
  dh::safe_cuda(cudaGetDevice(&device));
  return device;
}

// alternatively: `nvidia-smi -q | grep Addressing`
bool SupportsPageableMem() {
  std::int32_t res{0};
  dh::safe_cuda(cudaDeviceGetAttribute(&res, cudaDevAttrPageableMemoryAccess, CurrentDevice()));
  return res == 1;
}

bool SupportsAts() {
  std::int32_t res{0};
  dh::safe_cuda(cudaDeviceGetAttribute(&res, cudaDevAttrPageableMemoryAccessUsesHostPageTables,
                                       CurrentDevice()));
  return res == 1;
}

void CheckComputeCapability() {
  for (std::int32_t d_idx = 0; d_idx < AllVisibleGPUs(); ++d_idx) {
    cudaDeviceProp prop;
    dh::safe_cuda(cudaGetDeviceProperties(&prop, d_idx));
    std::ostringstream oss;
    oss << "CUDA Capability Major/Minor version number: " << prop.major << "." << prop.minor
        << " is insufficient.  Need >=3.5";
    int failed = prop.major < 3 || (prop.major == 3 && prop.minor < 5);
    if (failed) LOG(WARNING) << oss.str() << " for device: " << d_idx;
  }
}

void SetDevice(std::int32_t device) {
  if (device >= 0) {
    dh::safe_cuda(cudaSetDevice(device));
  }
}
#else
std::int32_t AllVisibleGPUs() { return 0; }

std::int32_t CurrentDevice() {
  AssertGPUSupport();
  return -1;
}

bool SupportsPageableMem() { return false; }

bool SupportsAts() { return false; }

void CheckComputeCapability() {}

void SetDevice(std::int32_t device) {
  if (device >= 0) {
    AssertGPUSupport();
  }
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace xgboost::common
