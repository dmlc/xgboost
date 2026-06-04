/**
 * Copyright 2026, XGBoost Contributors
 */
#pragma once

#if __has_include(<cuda/std/random>)
#define xgboost_HAS_CUDA_STD_RANDOM 1
#include <cuda/std/random>
#else
#include <thrust/random.h>
#endif

namespace xgboost::common::cuda_impl {
#if defined(xgboost_HAS_CUDA_STD_RANDOM)
using DefaultRng = cuda::std::philox4x64;
template <typename T>
using UniformRealDistribution = cuda::std::uniform_real_distribution<T>;
#else
using DefaultRng = thrust::default_random_engine;
template <typename T>
using UniformRealDistribution = thrust::uniform_real_distribution<T>;
#endif
}  // namespace xgboost::common::cuda_impl

#if defined(xgboost_HAS_CUDA_STD_RANDOM)
#undef xgboost_HAS_CUDA_STD_RANDOM
#endif
