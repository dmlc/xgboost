/**
 * Copyright 2024-2025, XGBoost contributors
 */
#pragma once
#include <cstddef>  // for size_t
#include <cstdint>  // for int32_t

#if defined(XGBOOST_USE_NVTX)
#include <nvtx3/nvtx3.hpp>
#endif  // defined(XGBOOST_USE_NVTX)

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

struct NvtxDomain {
  static constexpr char const *name{"libxgboost"};  // NOLINT
};

#if defined(XGBOOST_USE_NVTX)
using NvtxScopedRange = ::nvtx3::scoped_range_in<NvtxDomain>;
using NvtxEventAttr = ::nvtx3::event_attributes;
using NvtxRgb = ::nvtx3::rgb;
#else
class NvtxScopedRange {
 public:
  template <typename... Args>
  explicit NvtxScopedRange(Args &&...) {}
};
class NvtxEventAttr {
 public:
  template <typename... Args>
  explicit NvtxEventAttr(Args &&...) {}
};
class NvtxRgb {
 public:
  template <typename... Args>
  explicit NvtxRgb(Args &&...) {}
};
#endif  // defined(XGBOOST_USE_NVTX)
}  // namespace xgboost::curt

#if defined(XGBOOST_USE_NVTX)
#define xgboost_NVTX_FN_RANGE() NVTX3_FUNC_RANGE_IN(::xgboost::curt::NvtxDomain)
#else
#define xgboost_NVTX_FN_RANGE()
#endif  // defined(XGBOOST_USE_NVTX)
