/**
 * Copyright 2024-2025, XGBoost contributors
 */
#include "../common/cuda_dr_utils.h"  // for GetVersionFromSmi
#include "device_helpers.cuh"
#include "device_vector.cuh"  // for GrowOnlyVirtualMemVec
#include "xgboost/windefs.h"  // for xgboost_IS_WIN

namespace dh {
namespace {
[[nodiscard]] bool IsSupportedDrVer(std::int32_t major, std::int32_t minor) {
  return major > 12 || (major == 12 && minor >= 5);
}

// Check whether cuda virtual memory can be used.
// Host NUMA allocation requires driver that supports CTK >= 12.5 to be stable
[[nodiscard]] bool CheckVmAlloc() {
  std::int32_t major{0}, minor{0};
  xgboost::curt::GetDrVersionGlobal(&major, &minor);

  bool vm_flag = true;
  if (IsSupportedDrVer(major, minor)) {
    // The result from the driver api is not reliable. The system driver might not match
    // the CUDA driver in some obscure cases.
    //
    // https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
    // Ver                 Linux       Win
    // CUDA 12.5 Update 1  >=555.42.06 >=555.85
    // CUDA 12.5 GA        >=555.42.02 >=555.85
    vm_flag = xgboost::cudr::GetVersionFromSmiGlobal(&major, &minor) && major >= 555;
  } else {
    vm_flag = false;
  }
  return vm_flag;
}
}  // namespace

PinnedMemory::PinnedMemory() {
#if defined(xgboost_IS_WIN)
  this->impl_.emplace<detail::GrowOnlyPinnedMemoryImpl>();
#else
  if (CheckVmAlloc()) {
    this->impl_.emplace<detail::GrowOnlyVirtualMemVec>(CU_MEM_LOCATION_TYPE_HOST_NUMA);
  } else {
    this->impl_.emplace<detail::GrowOnlyPinnedMemoryImpl>();
  }
#endif
}
}  // namespace dh
