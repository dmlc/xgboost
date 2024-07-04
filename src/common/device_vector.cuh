/**
 * Copyright 2017-2024, XGBoost Contributors
 */
#pragma once
#include <thrust/device_malloc_allocator.h>  // for device_malloc_allocator
#include <thrust/device_ptr.h>               // for device_ptr
#include <thrust/device_vector.h>            // for device_vector

#if defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1
#include <rmm/device_uvector.hpp>                      // for device_uvector
#include <rmm/exec_policy.hpp>                         // for exec_policy_nosync
#include <rmm/mr/device/device_memory_resource.hpp>    // for device_memory_resource
#include <rmm/mr/device/per_device_resource.hpp>       // for get_current_device_resource
#include <rmm/mr/device/thrust_allocator_adaptor.hpp>  // for thrust_allocator
#include <rmm/version_config.hpp>                      // for RMM_VERSION_MAJOR

#include "xgboost/global_config.h"  // for GlobalConfigThreadLocalStore

#if !defined(RMM_VERSION_MAJOR) || !defined(RMM_VERSION_MINOR)

#error "Please use RMM version 0.18 or later"
#elif RMM_VERSION_MAJOR == 0 && RMM_VERSION_MINOR < 18
#error "Please use RMM version 0.18 or later"
#endif  // !defined(RMM_VERSION_MAJOR) || !defined(RMM_VERSION_MINOR)

#endif  // defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1

#include <cstddef>                 // for size_t
#include <cub/util_allocator.cuh>  // for CachingDeviceAllocator
#include <cub/util_device.cuh>     // for CurrentDevice
#include <map>                     // for map
#include <memory>                  // for unique_ptr

#include "common.h"  // for safe_cuda
#include "xgboost/logging.h"

namespace dh {
namespace detail {
/** \brief Keeps track of global device memory allocations. Thread safe.*/
class MemoryLogger {
  // Information for a single device
  struct DeviceStats {
    std::size_t currently_allocated_bytes{0};
    size_t peak_allocated_bytes{0};
    size_t num_allocations{0};
    size_t num_deallocations{0};
    std::map<void *, size_t> device_allocations;
    void RegisterAllocation(void *ptr, size_t n) {
      device_allocations[ptr] = n;
      currently_allocated_bytes += n;
      peak_allocated_bytes = std::max(peak_allocated_bytes, currently_allocated_bytes);
      num_allocations++;
      CHECK_GT(num_allocations, num_deallocations);
    }
    void RegisterDeallocation(void *ptr, size_t n, int current_device) {
      auto itr = device_allocations.find(ptr);
      if (itr == device_allocations.end()) {
        LOG(WARNING) << "Attempting to deallocate " << n << " bytes on device " << current_device
                     << " that was never allocated\n"
                     << dmlc::StackTrace();
      } else {
        num_deallocations++;
        CHECK_LE(num_deallocations, num_allocations);
        currently_allocated_bytes -= itr->second;
        device_allocations.erase(itr);
      }
    }
  };
  DeviceStats stats_;
  std::mutex mutex_;

 public:
  void RegisterAllocation(void *ptr, size_t n) {
    if (!xgboost::ConsoleLogger::ShouldLog(xgboost::ConsoleLogger::LV::kDebug)) {
      return;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    stats_.RegisterAllocation(ptr, n);
  }
  void RegisterDeallocation(void *ptr, size_t n) {
    if (!xgboost::ConsoleLogger::ShouldLog(xgboost::ConsoleLogger::LV::kDebug)) {
      return;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    stats_.RegisterDeallocation(ptr, n, cub::CurrentDevice());
  }
  size_t PeakMemory() const { return stats_.peak_allocated_bytes; }
  size_t CurrentlyAllocatedBytes() const { return stats_.currently_allocated_bytes; }
  void Clear() { stats_ = DeviceStats(); }

  void Log() {
    if (!xgboost::ConsoleLogger::ShouldLog(xgboost::ConsoleLogger::LV::kDebug)) {
      return;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    int current_device;
    dh::safe_cuda(cudaGetDevice(&current_device));
    LOG(CONSOLE) << "======== Device " << current_device << " Memory Allocations: "
                 << " ========";
    LOG(CONSOLE) << "Peak memory usage: " << stats_.peak_allocated_bytes / 1048576 << "MiB";
    LOG(CONSOLE) << "Number of allocations: " << stats_.num_allocations;
  }
};

void ThrowOOMError(std::string const &err, size_t bytes);
}  // namespace detail

inline detail::MemoryLogger &GlobalMemoryLogger() {
  static detail::MemoryLogger memory_logger;
  return memory_logger;
}

namespace detail {
#if defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1
template <typename T>
using XGBBaseDeviceAllocator = rmm::mr::thrust_allocator<T>;
#else   // defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1
template <typename T>
using XGBBaseDeviceAllocator = thrust::device_malloc_allocator<T>;
#endif  // defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1

/**
 * \brief Default memory allocator, uses cudaMalloc/Free and logs allocations if verbose.
 */
template <class T>
struct XGBDefaultDeviceAllocatorImpl : XGBBaseDeviceAllocator<T> {
  using SuperT = XGBBaseDeviceAllocator<T>;
  using pointer = thrust::device_ptr<T>;  // NOLINT
  template <typename U>
  struct rebind  // NOLINT
  {
    using other = XGBDefaultDeviceAllocatorImpl<U>;  // NOLINT
  };
  pointer allocate(size_t n) {  // NOLINT
    pointer ptr;
    try {
      ptr = SuperT::allocate(n);
      dh::safe_cuda(cudaGetLastError());
    } catch (const std::exception &e) {
      detail::ThrowOOMError(e.what(), n * sizeof(T));
    }
    GlobalMemoryLogger().RegisterAllocation(ptr.get(), n * sizeof(T));
    return ptr;
  }
  void deallocate(pointer ptr, size_t n) {  // NOLINT
    GlobalMemoryLogger().RegisterDeallocation(ptr.get(), n * sizeof(T));
    SuperT::deallocate(ptr, n);
  }
#if defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1
  XGBDefaultDeviceAllocatorImpl()
      : SuperT(rmm::cuda_stream_per_thread, rmm::mr::get_current_device_resource()) {}
#endif  // defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1
};

/**
 * \brief Caching memory allocator, uses cub::CachingDeviceAllocator as a back-end, unless
 *        RMM pool allocator is enabled. Does not initialise memory on construction.
 */
template <class T>
struct XGBCachingDeviceAllocatorImpl : XGBBaseDeviceAllocator<T> {
  using SuperT = XGBBaseDeviceAllocator<T>;
  using pointer = thrust::device_ptr<T>;  // NOLINT
  template <typename U>
  struct rebind  // NOLINT
  {
    using other = XGBCachingDeviceAllocatorImpl<U>;  // NOLINT
  };
  cub::CachingDeviceAllocator &GetGlobalCachingAllocator() {
    // Configure allocator with maximum cached bin size of ~1GB and no limit on
    // maximum cached bytes
    thread_local std::unique_ptr<cub::CachingDeviceAllocator> allocator{
        std::make_unique<cub::CachingDeviceAllocator>(2, 9, 29)};
    return *allocator;
  }
  pointer allocate(size_t n) {  // NOLINT
    pointer thrust_ptr;
    if (use_cub_allocator_) {
      T *raw_ptr{nullptr};
      auto errc = GetGlobalCachingAllocator().DeviceAllocate(reinterpret_cast<void **>(&raw_ptr),
                                                             n * sizeof(T));
      if (errc != cudaSuccess) {
        detail::ThrowOOMError("Caching allocator", n * sizeof(T));
      }
      thrust_ptr = pointer(raw_ptr);
    } else {
      try {
        thrust_ptr = SuperT::allocate(n);
        dh::safe_cuda(cudaGetLastError());
      } catch (const std::exception &e) {
        detail::ThrowOOMError(e.what(), n * sizeof(T));
      }
    }
    GlobalMemoryLogger().RegisterAllocation(thrust_ptr.get(), n * sizeof(T));
    return thrust_ptr;
  }
  void deallocate(pointer ptr, size_t n) {  // NOLINT
    GlobalMemoryLogger().RegisterDeallocation(ptr.get(), n * sizeof(T));
    if (use_cub_allocator_) {
      GetGlobalCachingAllocator().DeviceFree(ptr.get());
    } else {
      SuperT::deallocate(ptr, n);
    }
  }
#if defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1
  XGBCachingDeviceAllocatorImpl()
      : SuperT(rmm::cuda_stream_per_thread, rmm::mr::get_current_device_resource()),
        use_cub_allocator_(!xgboost::GlobalConfigThreadLocalStore::Get()->use_rmm) {}
#endif                                   // defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1
  XGBOOST_DEVICE void construct(T *) {}  // NOLINT
 private:
  bool use_cub_allocator_{true};
};
}  // namespace detail

// Declare xgboost allocators
// Replacement of allocator with custom backend should occur here
template <typename T>
using XGBDeviceAllocator = detail::XGBDefaultDeviceAllocatorImpl<T>;

/** Be careful that the initialization constructor is a no-op, which means calling
 *  `vec.resize(n)` won't initialize the memory region to 0. Instead use
 * `vec.resize(n, 0)`
 */
template <typename T>
using XGBCachingDeviceAllocator = detail::XGBCachingDeviceAllocatorImpl<T>;

/** @brief Specialisation of thrust device vector using custom allocator. */
template <typename T>
using device_vector = thrust::device_vector<T,  XGBDeviceAllocator<T>>;  // NOLINT
template <typename T>
using caching_device_vector = thrust::device_vector<T,  XGBCachingDeviceAllocator<T>>;  // NOLINT

#if defined(XGBOOST_USE_RMM)
/**
 * @brief Similar to `rmm::logging_resource_adaptor`, but uses XGBoost memory logger instead.
 */
class LoggingResource : public rmm::mr::device_memory_resource {
  rmm::mr::device_memory_resource *mr_{rmm::mr::get_current_device_resource()};

 public:
  LoggingResource() = default;
  ~LoggingResource() override = default;
  LoggingResource(LoggingResource const &) = delete;
  LoggingResource &operator=(LoggingResource const &) = delete;
  LoggingResource(LoggingResource &&) noexcept = default;
  LoggingResource &operator=(LoggingResource &&) noexcept = default;

  [[nodiscard]] rmm::device_async_resource_ref get_upstream_resource() const noexcept {  // NOLINT
    return mr_;
  }
  [[nodiscard]] rmm::mr::device_memory_resource *get_upstream() const noexcept {  // NOLINT
    return mr_;
  }

  void *do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override {  // NOLINT
    try {
      auto const ptr = mr_->allocate(bytes, stream);
      GlobalMemoryLogger().RegisterAllocation(ptr, bytes);
      return ptr;
    } catch (rmm::bad_alloc const &e) {
      detail::ThrowOOMError(e.what(), bytes);
    }
    return nullptr;
  }

  void do_deallocate(void *ptr, std::size_t bytes,  // NOLINT
                     rmm::cuda_stream_view stream) override {
    mr_->deallocate(ptr, bytes, stream);
    GlobalMemoryLogger().RegisterDeallocation(ptr, bytes);
  }

  [[nodiscard]] bool do_is_equal(  // NOLINT
      device_memory_resource const &other) const noexcept override {
    if (this == &other) {
      return true;
    }
    auto const *cast = dynamic_cast<LoggingResource const *>(&other);
    if (cast == nullptr) {
      return mr_->is_equal(other);
    }
    return get_upstream_resource() == cast->get_upstream_resource();
  }
};

LoggingResource *GlobalLoggingResource();

/**
 * @brief Container class that doesn't initialize the data.
 */
template <typename T>
class DeviceUVector : public rmm::device_uvector<T> {
  using Super = rmm::device_uvector<T>;

 public:
  DeviceUVector() : Super{0, rmm::cuda_stream_per_thread, GlobalLoggingResource()} {}

  void Resize(std::size_t n) { Super::resize(n, rmm::cuda_stream_per_thread); }
  void Resize(std::size_t n, T const &v) {
    auto orig = this->size();
    Super::resize(n, rmm::cuda_stream_per_thread);
    if (orig < n) {
      thrust::fill(rmm::exec_policy_nosync{}, this->begin() + orig, this->end(), v);
    }
  }

 private:
  // undefined private, cannot be accessed.
  void resize(std::size_t n, rmm::cuda_stream_view stream);  // NOLINT
};

#else

/**
 * @brief Without RMM, the initialization will happen.
 */
template <typename T>
class DeviceUVector : public thrust::device_vector<T, XGBDeviceAllocator<T>> {
  using Super = thrust::device_vector<T, XGBDeviceAllocator<T>>;

 public:
  void Resize(std::size_t n) { Super::resize(n); }
  void Resize(std::size_t n, T const &v) { Super::resize(n, v); }

 private:
  // undefined private, cannot be accessed.
  void resize(std::size_t n, T const &v = T{});  // NOLINT
};

#endif  // defined(XGBOOST_USE_RMM)
}  // namespace dh
