/**
 * Copyright 2022-2024, XGBoost Contributors
 *
 * @brief cuda pinned allocator for usage with thrust containers
 */

#pragma once

#include <cuda_runtime.h>

#include <cstddef>  // for size_t
#include <limits>   // for numeric_limits

#include "common.h"

namespace xgboost::common::cuda_impl {
// \p pinned_allocator is a CUDA-specific host memory allocator
//  that employs \c cudaMallocHost for allocation.
//
// This implementation is ported from the experimental/pinned_allocator
// that Thrust used to provide.
//
//  \see https://en.cppreference.com/w/cpp/memory/allocator

template <typename T>
struct PinnedAllocPolicy {
  using pointer = T*;              // NOLINT: The type returned by address() / allocate()
  using const_pointer = const T*;  // NOLINT: The type returned by address()
  using size_type = std::size_t;   // NOLINT: The type used for the size of the allocation
  using value_type = T;            // NOLINT: The type of the elements in the allocator

  size_type max_size() const {  // NOLINT
    return std::numeric_limits<size_type>::max() / sizeof(value_type);
  }

  pointer allocate(size_type cnt, const_pointer = nullptr) {  // NOLINT
    if (cnt > this->max_size()) {
      throw std::bad_alloc{};
    }  // end if

    pointer result(nullptr);
    dh::safe_cuda(cudaMallocHost(reinterpret_cast<void**>(&result), cnt * sizeof(value_type)));
    return result;
  }

  void deallocate(pointer p, size_type) { dh::safe_cuda(cudaFreeHost(p)); }  // NOLINT
};

template <typename T>
struct ManagedAllocPolicy {
  using pointer = T*;              // NOLINT: The type returned by address() / allocate()
  using const_pointer = const T*;  // NOLINT: The type returned by address()
  using size_type = std::size_t;   // NOLINT: The type used for the size of the allocation
  using value_type = T;            // NOLINT: The type of the elements in the allocator

  size_type max_size() const {  // NOLINT
    return std::numeric_limits<size_type>::max() / sizeof(value_type);
  }

  pointer allocate(size_type cnt, const_pointer = nullptr) {  // NOLINT
    if (cnt > this->max_size()) {
      throw std::bad_alloc{};
    }  // end if

    pointer result(nullptr);
    dh::safe_cuda(cudaMallocManaged(reinterpret_cast<void**>(&result), cnt * sizeof(value_type)));
    return result;
  }

  void deallocate(pointer p, size_type) { dh::safe_cuda(cudaFree(p)); }  // NOLINT
};

template <typename T, template <typename> typename Policy>
class CudaHostAllocatorImpl : public Policy<T> {  // NOLINT
 public:
  using value_type = typename Policy<T>::value_type;        // NOLINT
  using pointer = typename Policy<T>::pointer;              // NOLINT
  using const_pointer = typename Policy<T>::const_pointer;  // NOLINT
  using size_type = typename Policy<T>::size_type;          // NOLINT

  using reference = T&;              // NOLINT: The parameter type for address()
  using const_reference = const T&;  // NOLINT: The parameter type for address()

  using difference_type = std::ptrdiff_t;  // NOLINT: The type of the distance between two pointers

  template <typename U>
  struct rebind {                                    // NOLINT
    using other = CudaHostAllocatorImpl<U, Policy>;  // NOLINT: The rebound type
  };

  CudaHostAllocatorImpl() = default;
  ~CudaHostAllocatorImpl() = default;
  CudaHostAllocatorImpl(CudaHostAllocatorImpl const&) = default;

  CudaHostAllocatorImpl& operator=(CudaHostAllocatorImpl const& that) = default;
  CudaHostAllocatorImpl& operator=(CudaHostAllocatorImpl&& that) = default;

  template <typename U>
  CudaHostAllocatorImpl(CudaHostAllocatorImpl<U, Policy> const&) {}  // NOLINT

  pointer address(reference r) { return &r; }              // NOLINT
  const_pointer address(const_reference r) { return &r; }  // NOLINT

  bool operator==(CudaHostAllocatorImpl const& x) const { return true; }

  bool operator!=(CudaHostAllocatorImpl const& x) const { return !operator==(x); }
};

template <typename T>
using pinned_allocator = CudaHostAllocatorImpl<T, PinnedAllocPolicy>;  // NOLINT

template <typename T>
using managed_allocator = CudaHostAllocatorImpl<T, ManagedAllocPolicy>;  // NOLINT
}  // namespace xgboost::common::cuda_impl
