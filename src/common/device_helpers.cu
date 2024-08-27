/**
 * Copyright 2024, XGBoost contributors
 */
#include "cuda_rt_utils.h"
#include "device_helpers.cuh"

namespace dh {
PinnedMemory::PinnedMemory() {
  std::int32_t major{0}, minor{0};
  xgboost::curt::RtVersion(&major, &minor);
  // Host NUMA allocation requires >= 12.5 to be stable.
  if (major >= 12 && minor >= 5) {
    this->impl_.emplace<detail::GrowOnlyVirtualMemVec>(CU_MEM_LOCATION_TYPE_HOST_NUMA);
  } else {
    this->impl_.emplace<detail::GrowOnlyPinnedMemoryImpl>();
  }
}
}  // namespace dh
