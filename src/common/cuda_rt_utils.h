/**
 * Copyright 2024, XGBoost contributors
 */
#pragma once
#include <cstdint>  // for int32_t
namespace xgboost::common {
std::int32_t AllVisibleGPUs();

std::int32_t CurrentDevice();

// Whether the device supports coherently accessing pageable memory without calling
// `cudaHostRegister` on it
bool SupportsPageableMem();

// Address Translation Service (ATS)
bool SupportsAts();

void CheckComputeCapability();

void SetDevice(std::int32_t device);
}  // namespace xgboost::common
