/**
 * Copyright 2022-2024, XGBoost Contributors
 *
 * @brief cuda pinned allocator for usage with thrust containers
 */

#pragma once

#include <cuda_runtime.h>

#include <cstddef>  // for size_t
#include <limits>   // for numeric_limits
#include <new>      // for bad_array_new_length

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

  [[nodiscard]] constexpr size_type max_size() const {  // NOLINT
    return std::numeric_limits<size_type>::max() / sizeof(value_type);
  }

  [[nodiscard]] pointer allocate(size_type cnt, const_pointer = nullptr) const {  // NOLINT
    if (cnt > this->max_size()) {
      throw std::bad_array_new_length{};
    }

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

  [[nodiscard]] constexpr size_type max_size() const {  // NOLINT
    return std::numeric_limits<size_type>::max() / sizeof(value_type);
  }

  [[nodiscard]] pointer allocate(size_type cnt, const_pointer = nullptr) const {  // NOLINT
    if (cnt > this->max_size()) {
      throw std::bad_array_new_length{};
    }

    pointer result(nullptr);
    dh::safe_cuda(cudaMallocManaged(reinterpret_cast<void**>(&result), cnt * sizeof(value_type)));
    return result;
  }

  void deallocate(pointer p, size_type) { dh::safe_cuda(cudaFree(p)); }  // NOLINT
};

// This is actually a pinned memory allocator in disguise. We utilize HMM or ATS for
// efficient tracked memory allocation.
template <typename T>
struct SamAllocPolicy {
  using pointer = T*;              // NOLINT: The type returned by address() / allocate()
  using const_pointer = const T*;  // NOLINT: The type returned by address()
  using size_type = std::size_t;   // NOLINT: The type used for the size of the allocation
  using value_type = T;            // NOLINT: The type of the elements in the allocator

  [[nodiscard]] constexpr size_type max_size() const {  // NOLINT
    return std::numeric_limits<size_type>::max() / sizeof(value_type);
  }

  [[nodiscard]] pointer allocate(size_type cnt, const_pointer = nullptr) const {  // NOLINT
    if (cnt > this->max_size()) {
      throw std::bad_array_new_length{};
    }

    size_type n_bytes = cnt * sizeof(value_type);
    pointer result = reinterpret_cast<pointer>(std::malloc(n_bytes));
    if (!result) {
      throw std::bad_alloc{};
    }
    dh::safe_cuda(cudaHostRegister(result, n_bytes, cudaHostRegisterDefault));
    return result;
  }

  void deallocate(pointer p, size_type) {  // NOLINT
    dh::safe_cuda(cudaHostUnregister(p));
    std::free(p);
  }
};

template <typename T, template <typename> typename Policy>
class CudaHostAllocatorImpl : public Policy<T> {
 public:
  using typename Policy<T>::value_type;
  using typename Policy<T>::pointer;
  using typename Policy<T>::const_pointer;
  using typename Policy<T>::size_type;

  using reference = value_type&;              // NOLINT: The parameter type for address()
  using const_reference = const value_type&;  // NOLINT: The parameter type for address()

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

  bool operator==(CudaHostAllocatorImpl const&) const { return true; }

  bool operator!=(CudaHostAllocatorImpl const& x) const { return !operator==(x); }
};

template <typename T>
using PinnedAllocator = CudaHostAllocatorImpl<T, PinnedAllocPolicy>;

template <typename T>
using ManagedAllocator = CudaHostAllocatorImpl<T, ManagedAllocPolicy>;

template <typename T>
using SamAllocator = CudaHostAllocatorImpl<T, SamAllocPolicy>;
}  // namespace xgboost::common::cuda_impl
