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

#include <cuda.h>  // for CUmemGenericAllocationHandle

#include <atomic>                  // for atomic, memory_order
#include <cstddef>                 // for size_t
#include <cstdint>                 // for int64_t
#include <cub/util_allocator.cuh>  // for CachingDeviceAllocator
#include <cub/util_device.cuh>     // for CurrentDevice
#include <memory>                  // for unique_ptr

#include "common.h"         // for safe_cuda, HumanMemUnit
#include "cuda_dr_utils.h"  // for CuDriverApi
#include "xgboost/logging.h"
#include "xgboost/span.h"  // for Span

namespace dh {
namespace detail {
// std::atomic::fetch_max in c++26
template <typename T>
T AtomicFetchMax(std::atomic<T> &atom, T val,  // NOLINT
                 std::memory_order order = std::memory_order_seq_cst) {
  auto expected = atom.load();
  auto desired = expected > val ? expected : val;

  while (desired == val && !atom.compare_exchange_strong(expected, desired, order, order)) {
    desired = expected > val ? expected : val;
  }

  return expected;
}

/** \brief Keeps track of global device memory allocations. Thread safe.*/
class MemoryLogger {
  // Information for a single device
  struct DeviceStats {
    // Use signed int to allow temporary under-flow.
    std::atomic<std::int64_t> currently_allocated_bytes{0};
    std::atomic<std::int64_t> peak_allocated_bytes{0};
    void RegisterAllocation(std::int64_t n) {
      currently_allocated_bytes += n;
      AtomicFetchMax(peak_allocated_bytes, currently_allocated_bytes.load());
    }
    void RegisterDeallocation(std::int64_t n) { currently_allocated_bytes -= n; }
  };
  DeviceStats stats_;

 public:
  /**
   * @brief Register the allocation for logging.
   */
  void RegisterAllocation(std::size_t n) {
    if (!xgboost::ConsoleLogger::ShouldLog(xgboost::ConsoleLogger::LV::kDebug)) {
      return;
    }
    stats_.RegisterAllocation(static_cast<std::int64_t>(n));
  }
  /**
   * @brief Register the deallocation for logging.
   */
  void RegisterDeallocation(std::size_t n) {
    if (!xgboost::ConsoleLogger::ShouldLog(xgboost::ConsoleLogger::LV::kDebug)) {
      return;
    }
    stats_.RegisterDeallocation(static_cast<std::int64_t>(n));
  }
  std::int64_t PeakMemory() const { return stats_.peak_allocated_bytes; }
  std::int64_t CurrentlyAllocatedBytes() const { return stats_.currently_allocated_bytes; }
  void Clear() {
    stats_.currently_allocated_bytes = 0;
    stats_.peak_allocated_bytes = 0;
  }

  void Log() const {
    if (!xgboost::ConsoleLogger::ShouldLog(xgboost::ConsoleLogger::LV::kDebug)) {
      return;
    }
    auto current_device = cub::CurrentDevice();
    LOG(CONSOLE) << "======== Device " << current_device << " Memory Allocations: "
                 << " ========";
    LOG(CONSOLE) << "Peak memory usage: "
                 << xgboost::common::HumanMemUnit(stats_.peak_allocated_bytes);
  }
};

void ThrowOOMError(std::string const &err, std::size_t bytes);

struct GrowOnlyPinnedMemoryImpl {
  void *temp_storage{nullptr};
  size_t temp_storage_bytes{0};

  ~GrowOnlyPinnedMemoryImpl() { Free(); }

  template <typename T>
  xgboost::common::Span<T> GetSpan(size_t size) {
    size_t num_bytes = size * sizeof(T);
    if (num_bytes > temp_storage_bytes) {
      Free();
      safe_cuda(cudaMallocHost(&temp_storage, num_bytes));
      temp_storage_bytes = num_bytes;
    }
    return xgboost::common::Span<T>(static_cast<T *>(temp_storage), size);
  }

  void Free() {
    if (temp_storage != nullptr) {
      safe_cuda(cudaFreeHost(temp_storage));
    }
  }
};

/**
 * @brief Use low-level virtual memory functions from CUDA driver API for grow-only memory
 *        allocation.
 *
 * @url https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/
 *
 * Aside from the potential performance benefits, this is primarily implemented to prevent
 * deadlock in NCCL and XGBoost. The host NUMA version requires CTK12.5+ to be stable.
 */
class GrowOnlyVirtualMemVec {
  static auto RoundUp(std::size_t new_sz, std::size_t chunk_sz) {
    return ((new_sz + chunk_sz - 1) / chunk_sz) * chunk_sz;
  }

  struct PhyAddrHandle {
    CUmemGenericAllocationHandle handle;
    std::size_t size;
  };

  class VaRange {
    CUdeviceptr ptr_{0};
    std::size_t size_{0};

   public:
    VaRange(std::size_t size, CUdeviceptr hint, CUresult *p_status, bool *failed) : size_{size} {
      CUresult &status = *p_status;
      status = xgboost::cudr::GetGlobalCuDriverApi().cuMemAddressReserve(&ptr_, size, 0, hint, 0);
      *failed = status != CUDA_SUCCESS || (hint != 0 && ptr_ != hint);
    }
    ~VaRange() {
      if (ptr_ != 0) {
        xgboost::cudr::GetGlobalCuDriverApi().cuMemAddressFree(ptr_, this->size_);
      }
    }

    VaRange(VaRange const &that) = delete;
    VaRange &operator=(VaRange const &that) = delete;

    VaRange(VaRange &&that) { std::swap(*this, that); }
    VaRange &operator=(VaRange &&that) {
      std::swap(*this, that);
      return *this;
    }
    [[nodiscard]] auto DevPtr() const { return this->ptr_; }
    [[nodiscard]] std::size_t Size() const { return this->size_; }
  };

  using PhyHandle = std::unique_ptr<PhyAddrHandle, std::function<void(PhyAddrHandle *)>>;
  std::vector<PhyHandle> handles_;
  std::vector<std::unique_ptr<VaRange>> va_ranges_;

  xgboost::cudr::CuDriverApi &cu_{xgboost::cudr::GetGlobalCuDriverApi()};
  std::vector<CUmemAccessDesc> access_desc_;
  CUmemAllocationProp const prop_;

  // Always use bytes.
  std::size_t const granularity_;

  [[nodiscard]] std::size_t PhyCapacity() const;
  [[nodiscard]] CUdeviceptr DevPtr() const {
    if (this->va_ranges_.empty()) {
      return 0;
    }
    return this->va_ranges_.front()->DevPtr();
  }
  void MapBlock(CUdeviceptr ptr, PhyHandle const &hdl) const {
    safe_cu(cu_.cuMemMap(ptr, hdl->size, 0, hdl->handle, 0));
    safe_cu(cu_.cuMemSetAccess(ptr, hdl->size, access_desc_.data(), access_desc_.size()));
  }
  auto CreatePhysicalMem(std::size_t size) const {
    CUmemGenericAllocationHandle alloc_handle;
    auto padded_size = RoundUp(size, this->granularity_);
    safe_cu(this->cu_.cuMemCreate(&alloc_handle, padded_size, &this->prop_, 0));
    return alloc_handle;
  }
  void Reserve(std::size_t new_size);

 public:
  explicit GrowOnlyVirtualMemVec(CUmemLocationType type);

  void GrowTo(std::size_t n_bytes) {
    auto alloc_size = this->PhyCapacity();
    if (n_bytes <= alloc_size) {
      return;
    }

    std::size_t delta = n_bytes - alloc_size;
    auto const padded_delta = RoundUp(delta, this->granularity_);
    this->Reserve(alloc_size + padded_delta);

    this->handles_.emplace_back(
        std::unique_ptr<PhyAddrHandle, std::function<void(PhyAddrHandle *)>>{
            new PhyAddrHandle{this->CreatePhysicalMem(padded_delta), padded_delta}, [&](auto *hdl) {
              if (hdl) {
                cu_.cuMemRelease(hdl->handle);
              }
            }});
    auto ptr = this->DevPtr() + alloc_size;
    this->MapBlock(ptr, this->handles_.back());
  }

  template <typename T>
  xgboost::common::Span<T> GetSpan(std::size_t size) {
    size_t n_bytes = size * sizeof(T);
    this->GrowTo(n_bytes);
    return xgboost::common::Span<T>(reinterpret_cast<T *>(this->DevPtr()), size);
  }

  ~GrowOnlyVirtualMemVec() noexcept(false) {
    if (this->DevPtr() != 0) {
      safe_cu(cu_.cuMemUnmap(this->DevPtr(), this->PhyCapacity()));
    }

    this->va_ranges_.clear();  // make sure all VA are freed before releasing the handles.
    this->handles_.clear();    // release the handles
  }

  [[nodiscard]] void *data() { return reinterpret_cast<void *>(this->DevPtr()); }  // NOLINT
  [[nodiscard]] std::size_t size() const { return this->PhyCapacity(); }           // NOLINT
  [[nodiscard]] std::size_t Capacity() const;
};
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
    GlobalMemoryLogger().RegisterAllocation(n * sizeof(T));
    return ptr;
  }
  void deallocate(pointer ptr, size_t n) {  // NOLINT
    GlobalMemoryLogger().RegisterDeallocation(n * sizeof(T));
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
      // NOLINTBEGIN(clang-analyzer-unix.BlockInCriticalSection)
      auto errc = GetGlobalCachingAllocator().DeviceAllocate(reinterpret_cast<void **>(&raw_ptr),
                                                             n * sizeof(T));
      // NOLINTEND(clang-analyzer-unix.BlockInCriticalSection)
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
    GlobalMemoryLogger().RegisterAllocation(n * sizeof(T));
    return thrust_ptr;
  }
  void deallocate(pointer ptr, size_t n) {  // NOLINT
    GlobalMemoryLogger().RegisterDeallocation(n * sizeof(T));
    if (use_cub_allocator_) {
      GetGlobalCachingAllocator().DeviceFree(thrust::raw_pointer_cast(ptr));
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

/** @brief Specialisation of thrust device vector using custom allocator. In addition, it catches
 *         OOM errors.
 */
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
  LoggingResource(LoggingResource &&) noexcept = delete;
  LoggingResource &operator=(LoggingResource &&) noexcept = delete;

  [[nodiscard]] rmm::device_async_resource_ref get_upstream_resource() const noexcept {  // NOLINT
    return mr_;
  }
  [[nodiscard]] rmm::mr::device_memory_resource *get_upstream() const noexcept {  // NOLINT
    return mr_;
  }

  void *do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override {  // NOLINT
    try {
      auto const ptr = mr_->allocate(bytes, stream);
      GlobalMemoryLogger().RegisterAllocation(bytes);
      return ptr;
    } catch (rmm::bad_alloc const &e) {
      detail::ThrowOOMError(e.what(), bytes);
    }
    return nullptr;
  }

  void do_deallocate(void *ptr, std::size_t bytes,  // NOLINT
                     rmm::cuda_stream_view stream) override {
    mr_->deallocate(ptr, bytes, stream);
    GlobalMemoryLogger().RegisterDeallocation(bytes);
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

#endif  // defined(XGBOOST_USE_RMM)

/**
 * @brief Container class that doesn't initialize the data when RMM is used.
 */
template <typename T, bool is_caching>
class DeviceUVectorImpl {
 private:
#if defined(XGBOOST_USE_RMM)
  rmm::device_uvector<T> data_{0, rmm::cuda_stream_per_thread, GlobalLoggingResource()};
#else
  std::conditional_t<is_caching, ::dh::caching_device_vector<T>, ::dh::device_vector<T>> data_;
#endif  // defined(XGBOOST_USE_RMM)

 public:
  using value_type = T;                        // NOLINT
  using pointer = value_type *;                // NOLINT
  using const_pointer = value_type const *;    // NOLINT
  using reference = value_type &;              // NOLINT
  using const_reference = value_type const &;  // NOLINT

 public:
  DeviceUVectorImpl() = default;
  explicit DeviceUVectorImpl(std::size_t n) { this->resize(n); }
  DeviceUVectorImpl(DeviceUVectorImpl const &that) = delete;
  DeviceUVectorImpl &operator=(DeviceUVectorImpl const &that) = delete;
  DeviceUVectorImpl(DeviceUVectorImpl &&that) = default;
  DeviceUVectorImpl &operator=(DeviceUVectorImpl &&that) = default;

  void resize(std::size_t n) {  // NOLINT
#if defined(XGBOOST_USE_RMM)
    data_.resize(n, rmm::cuda_stream_per_thread);
#else
    data_.resize(n);
#endif
  }
  void resize(std::size_t n, T const &v) {         // NOLINT
#if defined(XGBOOST_USE_RMM)
    auto orig = this->size();
    data_.resize(n, rmm::cuda_stream_per_thread);
    if (orig < n) {
      thrust::fill(rmm::exec_policy_nosync{}, this->begin() + orig, this->end(), v);
    }
#else
    data_.resize(n, v);
#endif
  }

  void clear() {  // NOLINT
#if defined(XGBOOST_USE_RMM)
    this->data_.resize(0, rmm::cuda_stream_per_thread);
#else
    this->data_.clear();
#endif  // defined(XGBOOST_USE_RMM)
  }

  [[nodiscard]] std::size_t size() const { return data_.size(); }  // NOLINT
  [[nodiscard]] bool empty() const { return this->size() == 0; }   // NOLINT

  [[nodiscard]] auto begin() { return data_.begin(); }  // NOLINT
  [[nodiscard]] auto end() { return data_.end(); }      // NOLINT

  [[nodiscard]] auto begin() const { return this->cbegin(); }  // NOLINT
  [[nodiscard]] auto end() const { return this->cend(); }      // NOLINT

  [[nodiscard]] auto cbegin() const { return data_.cbegin(); }  // NOLINT
  [[nodiscard]] auto cend() const { return data_.cend(); }      // NOLINT

  [[nodiscard]] auto data() { return thrust::raw_pointer_cast(data_.data()); }        // NOLINT
  [[nodiscard]] auto data() const { return thrust::raw_pointer_cast(data_.data()); }  // NOLINT
};

template <typename T>
using DeviceUVector = DeviceUVectorImpl<T, false>;

template <typename T>
using CachingDeviceUVector = DeviceUVectorImpl<T, true>;
}  // namespace dh
