/**
 * Copyright 2024, XGBoost contributors
 *
 * @brief Utility for CUDA driver API.
 *
 * We don't link with libcuda.so at build time. The utilities here load the shared object
 * at runtime.
 */
#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>  // for int32_t

#include "xgboost/string_view.h"  // for StringView

namespace xgboost::cudr {
/**
 * @brief A struct for retrieving CUDA driver API from the runtime API.
 */
struct CuDriverApi {
  using Flags = unsigned long long;  // NOLINT

  using MemGetAllocationGranularityFn = CUresult(size_t *granularity,
                                                 const CUmemAllocationProp *prop,
                                                 CUmemAllocationGranularity_flags option);
  using MemCreateFn = CUresult(CUmemGenericAllocationHandle *handle, size_t size,
                               const CUmemAllocationProp *prop, Flags flags);
  using MemMapFn = CUresult(CUdeviceptr ptr, size_t size, size_t offset,
                            CUmemGenericAllocationHandle handle, Flags flags);
  using MemAddressReserveFn = CUresult(CUdeviceptr *ptr, size_t size, size_t alignment,
                                       CUdeviceptr addr, Flags flags);
  using MemSetAccessFn = CUresult(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc,
                                  size_t count);
  using MemUnmapFn = CUresult(CUdeviceptr ptr, size_t size);
  using MemReleaseFn = CUresult(CUmemGenericAllocationHandle handle);
  using MemAddressFreeFn = CUresult(CUdeviceptr ptr, size_t size);

  using GetErrorString = CUresult(CUresult error, const char **pStr);
  using GetErrorName = CUresult(CUresult error, const char **pStr);

  using DeviceGetAttribute = CUresult(int *pi, CUdevice_attribute attrib, CUdevice dev);
  using DeviceGet = CUresult(CUdevice *device, int ordinal);

  MemGetAllocationGranularityFn *cuMemGetAllocationGranularity{nullptr};  // NOLINT
  MemCreateFn *cuMemCreate{nullptr};                                      // NOLINT
  MemMapFn *cuMemMap{nullptr};                                            // NOLINT
  /**
   * @param[out] ptr       - Resulting pointer to start of virtual address range allocated
   * @param[in]  size      - Size of the reserved virtual address range requested
   * @param[in]  alignment - Alignment of the reserved virtual address range requested
   * @param[in]  addr      - Fixed starting address range requested
   * @param[in]  flags     - Currently unused, must be zero
   */
  MemAddressReserveFn *cuMemAddressReserve{nullptr};  // NOLINT
  MemSetAccessFn *cuMemSetAccess{nullptr};            // NOLINT
  MemUnmapFn *cuMemUnmap{nullptr};                    // NOLINT
  MemReleaseFn *cuMemRelease{nullptr};                // NOLINT
  MemAddressFreeFn *cuMemAddressFree{nullptr};        // NOLINT
  GetErrorString *cuGetErrorString{nullptr};          // NOLINT
  GetErrorName *cuGetErrorName{nullptr};              // NOLINT
  DeviceGetAttribute *cuDeviceGetAttribute{nullptr};  // NOLINT
  DeviceGet *cuDeviceGet{nullptr};                    // NOLINT

  CuDriverApi();

  void ThrowIfError(CUresult status, StringView fn, std::int32_t line, char const *file) const;
};

[[nodiscard]] CuDriverApi &GetGlobalCuDriverApi();

#define safe_cu(call)                                                                            \
  do {                                                                                           \
    auto __status = (call);                                                                      \
    if (__status != CUDA_SUCCESS) {                                                              \
      ::xgboost::cudr::GetGlobalCuDriverApi().ThrowIfError(__status, #call, __LINE__, __FILE__); \
    }                                                                                            \
  } while (0)

// Get the allocation granularity.
inline auto GetAllocGranularity(CUmemAllocationProp const *prop) {
  std::size_t granularity;
  safe_cu(GetGlobalCuDriverApi().cuMemGetAllocationGranularity(
      &granularity, prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  return granularity;
}

void GetCuLocation(CUmemLocationType type, CUmemLocation* loc);

// Describe the allocation property
[[nodiscard]] CUmemAllocationProp MakeAllocProp(CUmemLocationType type);
}  // namespace xgboost::cudr
