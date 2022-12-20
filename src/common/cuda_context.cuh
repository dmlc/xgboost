/**
 * Copyright 2022 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_CUDA_CONTEXT_CUH_
#define XGBOOST_COMMON_CUDA_CONTEXT_CUH_
#include <thrust/execution_policy.h>

#include "device_helpers.cuh"

namespace xgboost {
struct CUDAContext {
 private:
  dh::XGBCachingDeviceAllocator<char> caching_alloc_;
  dh::XGBDeviceAllocator<char> alloc_;

 public:
  /**
   * \brief Caching thrust policy.
   */
  auto CTP() const { return thrust::cuda::par(caching_alloc_).on(dh::DefaultStream()); }
  /**
   * \brief Thrust policy without caching allocator.
   */
  auto TP() const { return thrust::cuda::par(alloc_).on(dh::DefaultStream()); }
  auto Stream() const { return dh::DefaultStream(); }
};
}  // namespace xgboost
#endif  // XGBOOST_COMMON_CUDA_CONTEXT_CUH_
