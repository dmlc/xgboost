/**
 * Copyright 2024-2025, XGBoost contributors
 */
#pragma once
#include <cstddef>  // for size_t
#include <cstdint>  // for int32_t

namespace xgboost::curt {
std::int32_t AllVisibleGPUs();

/**
 * @param raise Raise error if XGBoost is not compiled with CUDA, or GPU is not available.
 */
std::int32_t CurrentDevice(bool raise = true);

// Whether the device supports coherently accessing pageable memory without calling
// `cudaHostRegister` on it
bool SupportsPageableMem();

// Address Translation Service (ATS)
bool SupportsAts();

void CheckComputeCapability();

void SetDevice(std::int32_t device);

/**
 * @brief Total device memory size.
 */
[[nodiscard]] std::size_t TotalMemory();

// Returns the CUDA Runtime version.
void RtVersion(std::int32_t* major, std::int32_t* minor);

// Returns the latest version of CUDA supported by the driver.
void DrVersion(std::int32_t* major, std::int32_t* minor);

// Get the current device's numa ID.
[[nodiscard]] std::int32_t GetNumaId();
}  // namespace xgboost::curt
