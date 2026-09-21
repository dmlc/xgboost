/**
 * Copyright 2026, XGBoost Contributors
 */
#pragma once

#include <thrust/version.h>  // for THRUST_VERSION

#if __has_include(<cuda/iterator>)
#include <cuda/iterator>  // for constant_iterator, counting_iterator
#else
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#endif  // __has_include(<cuda/iterator>)

#if THRUST_VERSION >= 300000
#include <cuda/functional>  // for maximum
#else
#include <thrust/functional.h>  // for maximum
#endif

namespace dh {
#if __has_include(<cuda/iterator>)
using cuda::counting_iterator;
using cuda::make_constant_iterator;
using cuda::make_counting_iterator;
#else
using thrust::counting_iterator;
using thrust::make_constant_iterator;
using thrust::make_counting_iterator;
#endif  // __has_include(<cuda/iterator>)

#if THRUST_VERSION >= 300000
using cuda::maximum;
#else
using thrust::maximum;
#endif
}  // namespace dh
