/**
 * Copyright 2017-2026, XGBoost Contributors
 */
#pragma once
#include <thrust/device_malloc_allocator.h>  // for device_malloc_allocator
#include <thrust/device_ptr.h>               // for device_ptr
#include <thrust/device_vector.h>            // for device_vector

#if defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1

#include <cuda/memory_resource>    // for async_resource_ref
#include <cuda/stream_ref>         // for stream_ref
#include <cuda/version>            // for CCCL_MAJOR_VERSION
#include <rmm/version_config.hpp>  // for RMM_VERSION_MAJOR

// TODO(hcho3): Remove this guard once we require Rapids 25.12+
#if (RMM_VERSION_MAJOR == 25 && RMM_VERSION_MINOR == 12) || RMM_VERSION_MAJOR >= 26
#include <rmm/mr/per_device_resource.hpp>  // for get_current_device_resource
#else  // (RMM_VERSION_MAJOR == 25 && RMM_VERSION_MINOR == 12) || RMM_VERSION_MAJOR >= 26
#include <rmm/mr/device/device_memory_resource.hpp>  // for device_memory_resource
#include <rmm/mr/device/per_device_resource.hpp>     // for get_current_device_resource
#endif  // (RMM_VERSION_MAJOR == 25 && RMM_VERSION_MINOR == 12) || RMM_VERSION_MAJOR >= 26

#else

#include "xgboost/windefs.h"  // for xgboost_IS_WIN

#endif  // defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1

#include <cuda.h>  // for CUmemGenericAllocationHandle

#include <atomic>                  // for atomic, memory_order
#include <cstddef>                 // for size_t
#include <cstdint>                 // for int64_t
#include <cub/util_allocator.cuh>  // for CachingDeviceAllocator
#include <cub/util_device.cuh>     // for CurrentDevice
#include <functional>              // for function
#include <memory>                  // for unique_ptr

#include "common.h"                 // for safe_cuda, HumanMemUnit
#include "cuda_dr_utils.h"          // for CuDriverApi
#include "cuda_stream.h"            // for DefaultStream
#include "xgboost/global_config.h"  // for GlobalConfigThreadLocalStore
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

#if defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1
using DeviceAsyncResourceRef = rmm::device_async_resource_ref;
#endif  // defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1

namespace detail {
#if defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1
/**
 * @brief Similar to `rmm::mr::thrust_allocator`.
 */
#ifdef __CUDACC__
#pragma nv_diagnostic push
#pragma nv_diag_suppress 20011
#endif
template <typename T>
class ThrustAllocMrAdapter {
// TODO(hcho3): Remove this guard once we require Rapids 25.12+
#if (RMM_VERSION_MAJOR == 25 && RMM_VERSION_MINOR == 12) || RMM_VERSION_MAJOR >= 26
#pragma nv_exec_check_disable
  DeviceAsyncResourceRef mr_{rmm::mr::get_current_device_resource_ref()};
#else   // (RMM_VERSION_MAJOR == 25 && RMM_VERSION_MINOR == 12) || RMM_VERSION_MAJOR >= 26
#pragma nv_exec_check_disable
  DeviceAsyncResourceRef mr_{rmm::mr::get_current_device_resource()};
#endif  // (RMM_VERSION_MAJOR == 25 && RMM_VERSION_MINOR == 12) || RMM_VERSION_MAJOR >= 26

 public:
  // NOLINTBEGIN
  using value_type = T;

  using pointer = thrust::device_ptr<T>;

  using const_pointer = thrust::device_ptr<const T>;

  using reference = thrust::device_reference<T>;

  using const_reference = thrust::device_reference<const T>;

  using size_type = std::size_t;

  using difference_type = typename pointer::difference_type;

  template <typename U>
  struct rebind {
    using other = ThrustAllocMrAdapter<U>;
  };

  pointer address(reference r) { return &r; }
  const_pointer address(const_reference r) { return &r; }

  pointer allocate(size_type n) {
    auto n_bytes = xgboost::common::SizeBytes<T>(n);
    auto s = cuda::stream_ref{::xgboost::curt::DefaultStream()};
#if (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1) || CCCL_MAJOR_VERSION > 3
    auto p = static_cast<T *>(mr_.allocate(s, n_bytes, std::alignment_of_v<T>));
#else   // (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1) || CCCL_MAJOR_VERSION > 3
    auto p = static_cast<T *>(mr_.allocate_async(n_bytes, std::alignment_of_v<T>, s));
#endif  // (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1) || CCCL_MAJOR_VERSION > 3
    return thrust::device_pointer_cast(p);
  }
  void deallocate(pointer ptr, size_type n) {
    auto n_bytes = xgboost::common::SizeBytes<T>(n);
    auto s = ::xgboost::curt::DefaultStream();
#if (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1) || CCCL_MAJOR_VERSION > 3
    return mr_.deallocate(cuda::stream_ref{s}, thrust::raw_pointer_cast(ptr), n_bytes);
#else   // (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1) || CCCL_MAJOR_VERSION > 3
    return mr_.deallocate_async(thrust::raw_pointer_cast(ptr), n_bytes, cuda::stream_ref{s});
#endif  // (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1) || CCCL_MAJOR_VERSION > 3
  }

  size_type max_size() const { return (::cuda::std::numeric_limits<size_type>::max)() / sizeof(T); }

  bool operator==(ThrustAllocMrAdapter const &) const { return true; }

  bool operator!=(ThrustAllocMrAdapter const &a) const { return !operator==(a); }

  // NOLINTEND
#pragma nv_exec_check_disable
  __host__ ThrustAllocMrAdapter() = default;
};
#ifdef __CUDACC__
#pragma nv_diagnostic pop
#endif

template <typename T>
using XGBBaseDeviceAllocator = ThrustAllocMrAdapter<T>;

#else  // defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1

/**
 * @brief Use CUDA async memory pool as an optional backing allocator.
 */
template <typename T>
class XGBAsyncPoolAllocator : public thrust::device_malloc_allocator<T> {
#if !defined(xgboost_IS_WIN)
  // MSVC/NVCC optimizes this variable away, as a result, we disable the async pool
  // entirely on Windows.
  std::int32_t use_async_pool_;
#endif

 public:
  using Super = thrust::device_malloc_allocator<T>;
  using pointer = typename Super::pointer;      // NOLINT(readability-identifier-naming)
  using size_type = typename Super::size_type;  // NOLINT(readability-identifier-naming)

#if defined(xgboost_IS_WIN)
  XGBAsyncPoolAllocator() = default;
#else
  XGBAsyncPoolAllocator()
      : use_async_pool_{::xgboost::GlobalConfigThreadLocalStore::Get()->use_cuda_async_pool} {}
#endif

  template <typename U>
  struct rebind {                            // NOLINT(readability-identifier-naming)
    using other = XGBAsyncPoolAllocator<U>;  // NOLINT(readability-identifier-naming)
  };

  pointer allocate(std::size_t n) {  // NOLINT
#if defined(xgboost_IS_WIN)
    return Super::allocate(n);
#else
    if (!this->use_async_pool_) {
      return Super::allocate(n);
    }

    T *raw_ptr = nullptr;
    auto n_bytes = xgboost::common::SizeBytes<T>(n);
    safe_cuda(cudaMallocAsync(&raw_ptr, n_bytes, xgboost::curt::DefaultStream()));
    return thrust::device_pointer_cast(raw_ptr);
#endif
  }

  void deallocate(pointer ptr, std::size_t n) {  // NOLINT
#if defined(xgboost_IS_WIN)
    return Super::deallocate(ptr, n);
#else
    if (!this->use_async_pool_) {
      return Super::deallocate(ptr, n);
    }

    safe_cuda(cudaFreeAsync(thrust::raw_pointer_cast(ptr), xgboost::curt::DefaultStream()));
#endif
  }

  // Used for tests.
  void SetAsync(bool use_async_pool) {
#if !defined(xgboost_IS_WIN)
    this->use_async_pool_ = use_async_pool;
#endif
  }
};

template <typename T>
using XGBBaseDeviceAllocator = XGBAsyncPoolAllocator<T>;
#endif  // defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1

/**
 * @brief Default memory allocator, uses cudaMalloc/Free and logs allocations if verbose.
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

  pointer allocate(std::size_t n) {  // NOLINT
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

  void deallocate(pointer ptr, std::size_t n) {  // NOLINT
    GlobalMemoryLogger().RegisterDeallocation(n * sizeof(T));
    SuperT::deallocate(ptr, n);
  }

  XGBDefaultDeviceAllocatorImpl() : SuperT{} {}
};

/**
 * @brief Caching memory allocator, uses cub::CachingDeviceAllocator as a back-end, unless
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

  static cub::CachingDeviceAllocator &GetGlobalCachingAllocator() {
    // Configure allocator with maximum cached bin size of ~1GB and no limit on
    // maximum cached bytes
    thread_local std::unique_ptr<cub::CachingDeviceAllocator> allocator{
        std::make_unique<cub::CachingDeviceAllocator>(2, 9, 29)};
    return *allocator;
  }

  pointer allocate(std::size_t n) {  // NOLINT
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
      thrust_ptr = thrust::device_pointer_cast(raw_ptr);
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

  void deallocate(pointer ptr, std::size_t n) {  // NOLINT
    if (use_cub_allocator_) {
      GetGlobalCachingAllocator().DeviceFree(thrust::raw_pointer_cast(ptr));
    } else {
      SuperT::deallocate(ptr, n);
    }
    GlobalMemoryLogger().RegisterDeallocation(n * sizeof(T));
  }

  XGBCachingDeviceAllocatorImpl()
      : SuperT{},
        use_cub_allocator_{!(xgboost::GlobalConfigThreadLocalStore::Get()->use_rmm ||
                             xgboost::GlobalConfigThreadLocalStore::Get()->use_cuda_async_pool)} {}

  XGBOOST_DEVICE void construct(T *) {}  // NOLINT

 private:
  bool use_cub_allocator_;
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
using device_vector = thrust::device_vector<T, XGBDeviceAllocator<T>>;  // NOLINT
template <typename T>
using caching_device_vector = thrust::device_vector<T, XGBCachingDeviceAllocator<T>>;  // NOLINT

/**
 * @brief Container class that doesn't initialize the data.
 */
template <typename T, bool is_caching>
class DeviceUVectorImpl {
 private:
  using Alloc =
      std::conditional_t<is_caching, dh::XGBCachingDeviceAllocator<T>, dh::XGBDeviceAllocator<T>>;
  Alloc alloc_;

  std::size_t size_{0};
  std::size_t capacity_{0};
  std::unique_ptr<T, std::function<void(T *)>> data_;

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

  [[nodiscard]] std::size_t Capacity() const { return this->capacity_; }

  // Resize without init.
  void resize(std::size_t n) {  // NOLINT
    using ::xgboost::common::SizeBytes;

    if (n <= this->Capacity()) {
      this->size_ = n;
      // early exit as no allocation is needed.
      return;
    }
    CHECK_LE(this->size(), this->Capacity());

    Alloc alloc = this->alloc_;
    decltype(data_) new_ptr{thrust::raw_pointer_cast(this->alloc_.allocate(n)),
                            [=](T *ptr) mutable {
                              if (ptr) {
                                alloc.deallocate(thrust::device_pointer_cast(ptr), n);
                              }
                            }};
    CHECK(new_ptr.get());

    auto s = ::xgboost::curt::DefaultStream();
    safe_cuda(cudaMemcpyAsync(new_ptr.get(), this->data(), SizeBytes<T>(this->size()),
                              cudaMemcpyDefault, s));
    this->size_ = n;
    this->capacity_ = n;

    this->data_ = std::move(new_ptr);
    // swap failed with CTK12.8
    // std::swap(this->data_, new_ptr);
  }
  // Resize with init
  void resize(std::size_t n, T const &v) {  // NOLINT
    auto orig = this->size();
    this->resize(n);
    if (orig < n) {
      auto exec = thrust::cuda::par_nosync.on(::xgboost::curt::DefaultStream());
      thrust::fill(exec, this->begin() + orig, this->end(), v);
    }
  }

  void clear() {  // NOLINT
    this->resize(0);
  }

  [[nodiscard]] std::size_t size() const { return this->size_; }  // NOLINT
  [[nodiscard]] bool empty() const { return this->size() == 0; }  // NOLINT

  [[nodiscard]] auto begin() { return this->data(); }               // NOLINT
  [[nodiscard]] auto end() { return this->data() + this->size(); }  // NOLINT

  [[nodiscard]] auto begin() const { return this->cbegin(); }  // NOLINT
  [[nodiscard]] auto end() const { return this->cend(); }      // NOLINT

  [[nodiscard]] auto cbegin() const { return this->data(); }               // NOLINT
  [[nodiscard]] auto cend() const { return this->data() + this->size(); }  // NOLINT

  [[nodiscard]] auto data() { return this->data_.get(); }        // NOLINT
  [[nodiscard]] auto data() const { return this->data_.get(); }  // NOLINT
};

template <typename T>
using DeviceUVector = DeviceUVectorImpl<T, false>;

template <typename T>
using CachingDeviceUVector = DeviceUVectorImpl<T, true>;
}  // namespace dh
