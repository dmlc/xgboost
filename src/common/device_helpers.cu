/**
 * Copyright 2024, XGBoost contributors
 */
#include "cuda_rt_utils.h"  // for RtVersion
#include "device_helpers.cuh"
#include "xgboost/windefs.h"  // for xgboost_IS_WIN

namespace dh {
PinnedMemory::PinnedMemory() {
#if defined(xgboost_IS_WIN) || !((CUDA_VERSION / 1000) >= 12 && ((CUDA_VERSION) % 100 / 10) >= 5)
  this->impl_.emplace<detail::GrowOnlyPinnedMemoryImpl>();
#else
  std::int32_t major{0}, minor{0};
  xgboost::curt::RtVersion(&major, &minor);
  // Host NUMA allocation requires >= 12.5 to be stable.
  if (major >= 12 && minor >= 5) {
    this->impl_.emplace<detail::GrowOnlyVirtualMemVec>(CU_MEM_LOCATION_TYPE_HOST_NUMA);
  } else {
    this->impl_.emplace<detail::GrowOnlyPinnedMemoryImpl>();
  }
#endif
}
}  // namespace dh
