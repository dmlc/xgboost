/**
 * Copyright 2024, XGBoost contributors
 */
#include "cuda_rt_utils.h"
#include "device_helpers.cuh"

namespace dh {
PinnedMemory::PinnedMemory() {
#if (defined(_MSC_VER) || defined(__MINGW32__)) || \
    !(__CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR__ >= 5)
  this->impl_.emplace<detail::GrowOnlyVirtualMemVec>(CU_MEM_LOCATION_TYPE_HOST_NUMA);
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
