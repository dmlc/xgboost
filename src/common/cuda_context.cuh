/**
 * Copyright 2022-2023, XGBoost Contributors
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
  auto CTP() const {
#if THRUST_MAJOR_VERSION >= 2 || defined(XGBOOST_USE_RMM)
    return thrust::cuda::par_nosync(caching_alloc_).on(dh::DefaultStream());
#else
    return thrust::cuda::par(caching_alloc_).on(dh::DefaultStream());
#endif  // THRUST_MAJOR_VERSION >= 2
  }
  /**
   * \brief Thrust policy without caching allocator.
   */
  auto TP() const {
#if THRUST_MAJOR_VERSION >= 2
    return thrust::cuda::par_nosync(alloc_).on(dh::DefaultStream());
#else
    return thrust::cuda::par(alloc_).on(dh::DefaultStream());
#endif  // THRUST_MAJOR_VERSION >= 2
  }
  auto Stream() const { return dh::DefaultStream(); }
};
}  // namespace xgboost
#endif  // XGBOOST_COMMON_CUDA_CONTEXT_CUH_
