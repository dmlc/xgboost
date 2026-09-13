/**
 * Copyright 2026, XGBoost Contributors
 */
#pragma once

#include <thrust/version.h>  // for THRUST_VERSION

#if THRUST_VERSION >= 300100
#include <cuda/iterator>  // for constant_iterator, counting_iterator
#else
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#endif

#if THRUST_VERSION >= 300000
#include <cuda/functional>  // for maximum
#else
#include <thrust/functional.h>  // for maximum
#endif

namespace dh {
// CUDA 12 and CUDA 13.0 bundle CCCL versions without <cuda/iterator>.
#if THRUST_VERSION >= 300100
using cuda::constant_iterator;
using cuda::counting_iterator;
using cuda::make_constant_iterator;
using cuda::make_counting_iterator;
#else
using thrust::constant_iterator;
using thrust::counting_iterator;
using thrust::make_constant_iterator;
using thrust::make_counting_iterator;
#endif

#if THRUST_VERSION >= 300000
using cuda::maximum;
#else
using thrust::maximum;
#endif
}  // namespace dh
