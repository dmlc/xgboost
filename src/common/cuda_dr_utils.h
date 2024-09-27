/**
 * Copyright 2024, XGBoost contributors
 *
 * @brief Utility for CUDA driver API.
 *
 * XGBoost doesn't link libcuda.so at build time. The utilities here load the shared
 * object at runtime.
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

  // Memroy manipulation functions.
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
  // Error handling
  using GetErrorString = CUresult(CUresult error, const char **pStr);
  using GetErrorName = CUresult(CUresult error, const char **pStr);
  // Device attributes
  using DeviceGetAttribute = CUresult(int *pi, CUdevice_attribute attrib, CUdevice dev);
  using DeviceGet = CUresult(CUdevice *device, int ordinal);

  MemGetAllocationGranularityFn *cuMemGetAllocationGranularity{nullptr};  // NOLINT
  MemCreateFn *cuMemCreate{nullptr};                                      // NOLINT
  /**
   * @param[in] offset - Must be zero.
   */
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

/**
 * @brief Macro for guarding CUDA driver API calls.
 */
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

/**
 * @brief Obtain appropriate device ordinal for `CUmemLocation`.
 */
void MakeCuMemLocation(CUmemLocationType type, CUmemLocation* loc);

/**
 * @brief Construct a `CUmemAllocationProp`.
 */
[[nodiscard]] CUmemAllocationProp MakeAllocProp(CUmemLocationType type);
}  // namespace xgboost::cudr
