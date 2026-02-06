/**
 * Copyright 2024-2026, XGBoost contributors
 */
#pragma once
#include <cstddef>  // for size_t
#include <cstdint>  // for int32_t
#include <string>   // for string

#include "cuda_stream.h"   // for StreamRef
#include "xgboost/span.h"  // for Span

namespace xgboost::curt {
std::int32_t AllVisibleGPUs();

/**
 * @param raise Raise error if XGBoost is not compiled with CUDA, or GPU is not available.
 */
std::int32_t CurrentDevice(bool raise = true);

// Whether the device supports coherently accessing pageable memory without calling
// `cudaHostRegister` on it
[[nodiscard]] bool SupportsPageableMem();

// Address Translation Service (ATS)
[[nodiscard]] bool SupportsAts();

void SetDevice(std::int32_t device);

/**
 * @brief Total device memory size.
 */
[[nodiscard]] std::size_t TotalMemory();

// Returns the CUDA Runtime version.
void GetRtVersionGlobal(std::int32_t* major, std::int32_t* minor);

// Returns the latest version of CUDA supported by the driver.
void GetDrVersionGlobal(std::int32_t* major, std::int32_t* minor);

// Get the current device's numa ID.
[[nodiscard]] std::int32_t GetNumaId();

[[nodiscard]] std::int32_t GetMpCnt(std::int32_t device);

[[nodiscard]] bool MemoryPoolsSupported(std::int32_t device);

inline constexpr std::size_t kUuidLength = 16;

void GetUuid(common::Span<unsigned char> uuid, std::int32_t device);

[[nodiscard]] std::string PrintUuid(common::Span<unsigned char const, kUuidLength> uuid);

// cudaMemcpyAsync
void MemcpyAsync(void* dst, const void* src, std::size_t count, StreamRef stream);
}  // namespace xgboost::curt
