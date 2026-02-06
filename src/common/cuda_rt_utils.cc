/**
 * Copyright 2015-2026, XGBoost Contributors
 */
#include "cuda_rt_utils.h"

#include "cuda_stream.h"   // for StreamRef
#include "xgboost/span.h"  // for Span

#if defined(XGBOOST_USE_CUDA)
#include <cuda_runtime_api.h>

#include <algorithm>  // for max

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
[[nodiscard]] bool SupportsPageableMem() {
  std::int32_t res{0};
  dh::safe_cuda(cudaDeviceGetAttribute(&res, cudaDevAttrPageableMemoryAccess, CurrentDevice()));
  return res == 1;
}

[[nodiscard]] bool SupportsAts() {
  std::int32_t res{0};
  dh::safe_cuda(cudaDeviceGetAttribute(&res, cudaDevAttrPageableMemoryAccessUsesHostPageTables,
                                       CurrentDevice()));
  return res == 1;
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

void GetRtVersionGlobal(std::int32_t* major, std::int32_t* minor) {
  GetVersionImpl([](std::int32_t* ver) { dh::safe_cuda(cudaRuntimeGetVersion(ver)); }, major,
                 minor);
}

void GetDrVersionGlobal(std::int32_t* major, std::int32_t* minor) {
  GetVersionImpl([](std::int32_t* ver) { dh::safe_cuda(cudaDriverGetVersion(ver)); }, major, minor);
}

[[nodiscard]] std::int32_t GetNumaId() {
  std::int32_t numa_id = -1;
  dh::safe_cuda(cudaDeviceGetAttribute(&numa_id, cudaDevAttrHostNumaId, curt::CurrentDevice()));
  numa_id = std::max(numa_id, 0);
  return numa_id;
}

[[nodiscard]] std::int32_t GetMpCnt(std::int32_t device) {
  std::int32_t n_mps = 0;
  dh::safe_cuda(cudaDeviceGetAttribute(&n_mps, cudaDevAttrMultiProcessorCount, device));
  CHECK_GT(n_mps, 0);
  return n_mps;
}

[[nodiscard]] bool MemoryPoolsSupported(std::int32_t device) {
  std::int32_t res = 0;
  dh::safe_cuda(cudaDeviceGetAttribute(&res, cudaDevAttrMemoryPoolsSupported, device));
  return !!res;
}

static_assert(kUuidLength == sizeof(std::declval<cudaDeviceProp>().uuid));

void GetUuid(xgboost::common::Span<unsigned char> uuid, std::int32_t device) {
  cudaDeviceProp prob{};
  dh::safe_cuda(cudaGetDeviceProperties(&prob, device));
  std::memcpy(uuid.data(), static_cast<void*>(&(prob.uuid)), kUuidLength);
}

[[nodiscard]] std::string PrintUuid(common::Span<unsigned char const, kUuidLength> uuid) {
  std::set<std::size_t> dash_pos{0, 4, 6, 8, 10};
  std::stringstream ss;
  ss << "GPU";
  for (std::size_t i = 0; i < kUuidLength; ++i) {
    if (dash_pos.find(i) != dash_pos.cend()) {
      ss << "-";
    }
    std::stringstream byte_ss;
    byte_ss << std::hex << (0xFF & std::uint32_t{uuid[i]});
    auto byte_str = byte_ss.str();
    if (byte_str.length() == 1) {
      byte_str = "0" + byte_str;
    };
    ss << byte_str;
  }
  return ss.str();
}

void MemcpyAsync(void* dst, const void* src, std::size_t count, StreamRef stream) {
  dh::safe_cuda(cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream));
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

[[nodiscard]] std::int32_t GetNumaId() {
  common::AssertGPUSupport();
  return 0;
}

[[nodiscard]] std::int32_t GetMpCnt(std::int32_t) {
  common::AssertGPUSupport();
  return 0;
}

[[nodiscard]] bool MemoryPoolsSupported(std::int32_t) { return false; }

void GetUuid(xgboost::common::Span<unsigned char>, std::int32_t) { common::AssertGPUSupport(); }

void MemcpyAsync(void*, const void*, std::size_t, StreamRef) { common::AssertGPUSupport(); }

#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace xgboost::curt
