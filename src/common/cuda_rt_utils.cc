/**
 * Copyright 2015-2025, XGBoost Contributors
 */
#include "cuda_rt_utils.h"

#if defined(XGBOOST_USE_CUDA)
#include <cuda_runtime_api.h>
#endif  // defined(XGBOOST_USE_CUDA)

#include <cstddef>  // for size_t
#include <cstdint>  // for int32_t
#include <mutex>    // for once_flag, call_once

#include "common.h"  // for safe_cuda

namespace xgboost::curt {
#if defined(XGBOOST_USE_CUDA)
std::int32_t AllVisibleGPUs() {
  int n_visgpus = 0;
  try {
    // When compiled with CUDA but running on CPU only device,
    // cudaGetDeviceCount will fail.
    dh::safe_cuda(cudaGetDeviceCount(&n_visgpus));
  } catch (const dmlc::Error&) {
    cudaGetLastError();  // reset error.
    return 0;
  }
  return n_visgpus;
}

std::int32_t CurrentDevice(bool raise) {
  std::int32_t device = -1;
  if (raise) {
    dh::safe_cuda(cudaGetDevice(&device));
  } else if (cudaGetDevice(&device) != cudaSuccess) {
    // Return -1 as an error.
    return -1;
  }
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

[[nodiscard]] std::size_t TotalMemory() {
  std::size_t device_free = 0;
  std::size_t device_total = 0;
  dh::safe_cuda(cudaMemGetInfo(&device_free, &device_total));
  return device_total;
}

namespace {
template <typename Fn>
void GetVersionImpl(Fn&& fn, std::int32_t* major, std::int32_t* minor) {
  static std::int32_t version = 0;
  static std::once_flag flag;
  std::call_once(flag, [&] { fn(&version); });
  if (major) {
    *major = version / 1000;
  }
  if (minor) {
    *minor = version % 100 / 10;
  }
}
}  // namespace

void RtVersion(std::int32_t* major, std::int32_t* minor) {
  GetVersionImpl([](std::int32_t* ver) { dh::safe_cuda(cudaRuntimeGetVersion(ver)); }, major,
                 minor);
}

void DrVersion(std::int32_t* major, std::int32_t* minor) {
  GetVersionImpl([](std::int32_t* ver) { dh::safe_cuda(cudaDriverGetVersion(ver)); }, major, minor);
}

#else
std::int32_t AllVisibleGPUs() { return 0; }

std::int32_t CurrentDevice(bool raise) {
  if (raise) {
    common::AssertGPUSupport();
  }
  return -1;
}

bool SupportsPageableMem() { return false; }

bool SupportsAts() { return false; }

[[nodiscard]] std::size_t TotalMemory() { return 0; }

void CheckComputeCapability() {}

void SetDevice(std::int32_t device) {
  if (device >= 0) {
    common::AssertGPUSupport();
  }
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace xgboost::curt
