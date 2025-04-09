/**
 * Copyright 2024-2025, XGBoost contributors
 */
#include <mutex>  // for once_flag, call_once

#include "cuda_rt_utils.h"  // for RtVersion
#include "device_helpers.cuh"
#include "device_vector.cuh"  // for GrowOnlyVirtualMemVec
#include "xgboost/windefs.h"  // for xgboost_IS_WIN

namespace dh {
namespace {
// Check whether cuda virtual memory can be used.
// Host NUMA allocation requires driver that supports CTK >= 12.5 to be stable
[[nodiscard]] bool CheckVmAlloc() {
  static bool vm_flag = true;
  static std::once_flag once;

  std::call_once(once, [] {
    std::int32_t major{0}, minor{0};
    xgboost::curt::DrVersion(&major, &minor);
    if (major > 12 || (major == 12 && minor >= 5)) {
      vm_flag = true;
    } else {
      vm_flag = false;
    }
  });
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
