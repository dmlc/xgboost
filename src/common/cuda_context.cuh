/**
 * Copyright 2022-2026, XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_CUDA_CONTEXT_CUH_
#define XGBOOST_COMMON_CUDA_CONTEXT_CUH_
#include <thrust/execution_policy.h>

#include "cuda_stream.h"      // for DefaultStream
#include "device_vector.cuh"  // for XGBCachingDeviceAllocator, XGBDeviceAllocator

namespace xgboost {
struct CUDAContext {
 private:
  dh::XGBCachingDeviceAllocator<char> caching_alloc_;
  dh::XGBDeviceAllocator<char> alloc_;

 public:
  /**
   * @brief Caching thrust policy.
   */
  auto CTP() const {
    return thrust::cuda::par_nosync(caching_alloc_).on(curt::DefaultStream());
  }
  /**
   * @brief Thrust policy without caching allocator.
   */
  auto TP() const {
    return thrust::cuda::par_nosync(alloc_).on(curt::DefaultStream());
  }
  auto Stream() const { return curt::DefaultStream(); }
};
}  // namespace xgboost
#endif  // XGBOOST_COMMON_CUDA_CONTEXT_CUH_
