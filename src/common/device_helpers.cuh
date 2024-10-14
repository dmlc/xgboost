/**
 * Copyright 2017-2023 XGBoost contributors
 */
#pragma once
#include <thrust/binary_search.h>  // thrust::upper_bound
#include <thrust/device_malloc_allocator.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>                    // thrust::seq
#include <thrust/gather.h>                              // gather
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>  // make_transform_output_iterator
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
#include <thrust/transform_scan.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cstddef>  // for size_t
#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "../collective/communicator-inl.h"
#include "common.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/logging.h"
#include "xgboost/span.h"

#if defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1
#include "rmm/mr/device/per_device_resource.hpp"
#include "rmm/mr/device/thrust_allocator_adaptor.hpp"
#include "rmm/version_config.hpp"

#if !defined(RMM_VERSION_MAJOR) || !defined(RMM_VERSION_MINOR)
#error "Please use RMM version 0.18 or later"
#elif RMM_VERSION_MAJOR == 0 && RMM_VERSION_MINOR < 18
#error "Please use RMM version 0.18 or later"
#endif  // !defined(RMM_VERSION_MAJOR) || !defined(RMM_VERSION_MINOR)

#endif  // defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600 || defined(__clang__)

#else  // In device code and CUDA < 600
__device__ __forceinline__ double atomicAdd(double* address, double val) {  // NOLINT
  unsigned long long int* address_as_ull =
      (unsigned long long int*)address;                   // NOLINT
  unsigned long long int old = *address_as_ull, assumed;  // NOLINT

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

namespace dh {

// FIXME(jiamingy): Remove this once we get rid of cub submodule.
constexpr bool BuildWithCUDACub() {
#if defined(THRUST_IGNORE_CUB_VERSION_CHECK) && THRUST_IGNORE_CUB_VERSION_CHECK == 1
  return false;
#else
  return true;
#endif // defined(THRUST_IGNORE_CUB_VERSION_CHECK) && THRUST_IGNORE_CUB_VERSION_CHECK == 1
}

namespace detail {
template <size_t size>
struct AtomicDispatcher;

template <>
struct AtomicDispatcher<sizeof(uint32_t)> {
  using Type = unsigned int;  // NOLINT
  static_assert(sizeof(Type) == sizeof(uint32_t), "Unsigned should be of size 32 bits.");
};

template <>
struct AtomicDispatcher<sizeof(uint64_t)> {
  using Type = unsigned long long;  // NOLINT
  static_assert(sizeof(Type) == sizeof(uint64_t), "Unsigned long long should be of size 64 bits.");
};
}  // namespace detail
}  // namespace dh

// atomicAdd is not defined for size_t.
template <typename T = size_t,
          std::enable_if_t<std::is_same<size_t, T>::value &&
                           !std::is_same<size_t, unsigned long long>::value> * =  // NOLINT
              nullptr>
XGBOOST_DEV_INLINE T atomicAdd(T *addr, T v) {  // NOLINT
  using Type = typename dh::detail::AtomicDispatcher<sizeof(T)>::Type;
  Type ret = ::atomicAdd(reinterpret_cast<Type *>(addr), static_cast<Type>(v));
  return static_cast<T>(ret);
}
namespace dh {

inline int32_t CudaGetPointerDevice(void const *ptr) {
  int32_t device = -1;
  cudaPointerAttributes attr;
  dh::safe_cuda(cudaPointerGetAttributes(&attr, ptr));
  device = attr.device;
  return device;
}

inline size_t AvailableMemory(int device_idx) {
  size_t device_free = 0;
  size_t device_total = 0;
  safe_cuda(cudaSetDevice(device_idx));
  dh::safe_cuda(cudaMemGetInfo(&device_free, &device_total));
  return device_free;
}

inline int32_t CurrentDevice() {
  int32_t device = 0;
  safe_cuda(cudaGetDevice(&device));
  return device;
}

inline size_t TotalMemory(int device_idx) {
  size_t device_free = 0;
  size_t device_total = 0;
  safe_cuda(cudaSetDevice(device_idx));
  dh::safe_cuda(cudaMemGetInfo(&device_free, &device_total));
  return device_total;
}

/**
 * \fn  inline int MaxSharedMemory(int device_idx)
 *
 * \brief Maximum shared memory per block on this device.
 *
 * \param device_idx  Zero-based index of the device.
 */

inline size_t MaxSharedMemory(int device_idx) {
  int max_shared_memory = 0;
  dh::safe_cuda(cudaDeviceGetAttribute
                (&max_shared_memory, cudaDevAttrMaxSharedMemoryPerBlock,
                 device_idx));
  return static_cast<std::size_t>(max_shared_memory);
}

/**
 * \fn  inline int MaxSharedMemoryOptin(int device_idx)
 *
 * \brief Maximum dynamic shared memory per thread block on this device
     that can be opted into when using cudaFuncSetAttribute().
 *
 * \param device_idx  Zero-based index of the device.
 */

inline size_t MaxSharedMemoryOptin(int device_idx) {
  int max_shared_memory = 0;
  dh::safe_cuda(cudaDeviceGetAttribute
                (&max_shared_memory, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                 device_idx));
  return static_cast<std::size_t>(max_shared_memory);
}

inline void CheckComputeCapability() {
  for (int d_idx = 0; d_idx < xgboost::common::AllVisibleGPUs(); ++d_idx) {
    cudaDeviceProp prop;
    safe_cuda(cudaGetDeviceProperties(&prop, d_idx));
    std::ostringstream oss;
    oss << "CUDA Capability Major/Minor version number: " << prop.major << "."
        << prop.minor << " is insufficient.  Need >=3.5";
    int failed = prop.major < 3 || (prop.major == 3 && prop.minor < 5);
    if (failed) LOG(WARNING) << oss.str() << " for device: " << d_idx;
  }
}

XGBOOST_DEV_INLINE void AtomicOrByte(unsigned int *__restrict__ buffer,
                                     size_t ibyte, unsigned char b) {
  atomicOr(&buffer[ibyte / sizeof(unsigned int)],
           static_cast<unsigned int>(b)
               << (ibyte % (sizeof(unsigned int)) * 8));
}

template <typename T>
__device__ xgboost::common::Range GridStrideRange(T begin, T end) {
  begin += blockDim.x * blockIdx.x + threadIdx.x;
  xgboost::common::Range r(begin, end);
  r.Step(gridDim.x * blockDim.x);
  return r;
}

template <typename T>
__device__ xgboost::common::Range BlockStrideRange(T begin, T end) {
  begin += threadIdx.x;
  xgboost::common::Range r(begin, end);
  r.Step(blockDim.x);
  return r;
}

// Threadblock iterates over range, filling with value. Requires all threads in
// block to be active.
template <typename IterT, typename ValueT>
__device__ void BlockFill(IterT begin, size_t n, ValueT value) {
  for (auto i : BlockStrideRange(static_cast<size_t>(0), n)) {
    begin[i] = value;
  }
}

/*
 * Kernel launcher
 */

template <typename L>
__global__ void LaunchNKernel(size_t begin, size_t end, L lambda) {
  for (auto i : GridStrideRange(begin, end)) {
    lambda(i);
  }
}

/* \brief A wrapper around kernel launching syntax, used to guard against empty input.
 *
 * - nvcc fails to deduce template argument when kernel is a template accepting __device__
 *   function as argument.  Hence functions like `LaunchN` cannot use this wrapper.
 *
 * - With c++ initialization list `{}` syntax, you are forced to comply with the CUDA type
 *   specification.
 */
class LaunchKernel {
  size_t shmem_size_;
  cudaStream_t stream_;

  dim3 grids_;
  dim3 blocks_;

 public:
  LaunchKernel(uint32_t _grids, uint32_t _blk, size_t _shmem=0, cudaStream_t _s=nullptr) :
      grids_{_grids, 1, 1}, blocks_{_blk, 1, 1}, shmem_size_{_shmem}, stream_{_s} {}
  LaunchKernel(dim3 _grids, dim3 _blk, size_t _shmem=0, cudaStream_t _s=nullptr) :
      grids_{_grids}, blocks_{_blk}, shmem_size_{_shmem}, stream_{_s} {}

  template <typename K, typename... Args>
  void operator()(K kernel, Args... args) {
    if (XGBOOST_EXPECT(grids_.x * grids_.y * grids_.z == 0, false)) {
      LOG(DEBUG) << "Skipping empty CUDA kernel.";
      return;
    }
    kernel<<<grids_, blocks_, shmem_size_, stream_>>>(args...);  // NOLINT
  }
};

template <int ITEMS_PER_THREAD = 8, int BLOCK_THREADS = 256, typename L>
inline void LaunchN(size_t n, cudaStream_t stream, L lambda) {
  if (n == 0) {
    return;
  }
  const int GRID_SIZE =
      static_cast<int>(xgboost::common::DivRoundUp(n, ITEMS_PER_THREAD * BLOCK_THREADS));
  LaunchNKernel<<<GRID_SIZE, BLOCK_THREADS, 0, stream>>>(  // NOLINT
      static_cast<size_t>(0), n, lambda);
}

// Default stream version
template <int ITEMS_PER_THREAD = 8, int BLOCK_THREADS = 256, typename L>
inline void LaunchN(size_t n, L lambda) {
  LaunchN<ITEMS_PER_THREAD, BLOCK_THREADS>(n, nullptr, lambda);
}

template <typename Container>
void Iota(Container array, cudaStream_t stream) {
  LaunchN(array.size(), stream, [=] __device__(size_t i) { array[i] = i; });
}

namespace detail {
/** \brief Keeps track of global device memory allocations. Thread safe.*/
class MemoryLogger {
  // Information for a single device
  struct DeviceStats {
    size_t currently_allocated_bytes{ 0 };
    size_t peak_allocated_bytes{ 0 };
    size_t num_allocations{ 0 };
    size_t num_deallocations{ 0 };
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
    int current_device;
    safe_cuda(cudaGetDevice(&current_device));
    stats_.RegisterAllocation(ptr, n);
  }
  void RegisterDeallocation(void *ptr, size_t n) {
    if (!xgboost::ConsoleLogger::ShouldLog(xgboost::ConsoleLogger::LV::kDebug)) {
      return;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    int current_device;
    safe_cuda(cudaGetDevice(&current_device));
    stats_.RegisterDeallocation(ptr, n, current_device);
  }
  size_t PeakMemory() const {
    return stats_.peak_allocated_bytes;
  }
  size_t CurrentlyAllocatedBytes() const {
    return stats_.currently_allocated_bytes;
  }
  void Clear()
  {
    stats_ = DeviceStats();
  }

  void Log() {
    if (!xgboost::ConsoleLogger::ShouldLog(xgboost::ConsoleLogger::LV::kDebug)) {
      return;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    int current_device;
    safe_cuda(cudaGetDevice(&current_device));
    LOG(CONSOLE) << "======== Device " << current_device << " Memory Allocations: "
      << " ========";
    LOG(CONSOLE) << "Peak memory usage: "
      << stats_.peak_allocated_bytes / 1048576 << "MiB";
    LOG(CONSOLE) << "Number of allocations: " << stats_.num_allocations;
  }
};
}  // namespace detail

inline detail::MemoryLogger &GlobalMemoryLogger() {
  static detail::MemoryLogger memory_logger;
  return memory_logger;
}

// dh::DebugSyncDevice(__FILE__, __LINE__);
inline void DebugSyncDevice(std::string file="", int32_t line = -1) {
  if (file != "" && line != -1) {
    auto rank = xgboost::collective::GetRank();
    LOG(DEBUG) << "R:" << rank << ": " << file << ":" << line;
  }
  safe_cuda(cudaDeviceSynchronize());
  safe_cuda(cudaGetLastError());
}

namespace detail {

#if defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1
template <typename T>
using XGBBaseDeviceAllocator = rmm::mr::thrust_allocator<T>;
#else  // defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1
template <typename T>
using XGBBaseDeviceAllocator = thrust::device_malloc_allocator<T>;
#endif  // defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1

inline void ThrowOOMError(std::string const& err, size_t bytes) {
  auto device = CurrentDevice();
  auto rank = xgboost::collective::GetRank();
  std::stringstream ss;
  ss << "Memory allocation error on worker " << rank << ": " << err << "\n"
     << "- Free memory: " << AvailableMemory(device) << "\n"
     << "- Requested memory: " << bytes << std::endl;
  LOG(FATAL) << ss.str();
}

/**
 * \brief Default memory allocator, uses cudaMalloc/Free and logs allocations if verbose.
 */
template <class T>
struct XGBDefaultDeviceAllocatorImpl : XGBBaseDeviceAllocator<T> {
  using SuperT = XGBBaseDeviceAllocator<T>;
  using pointer = thrust::device_ptr<T>;  // NOLINT
  template<typename U>
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
      ThrowOOMError(e.what(), n * sizeof(T));
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
    : SuperT(rmm::cuda_stream_default, rmm::mr::get_current_device_resource()) {}
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
  template<typename U>
  struct rebind  // NOLINT
  {
    using other = XGBCachingDeviceAllocatorImpl<U>;  // NOLINT
  };
  cub::CachingDeviceAllocator& GetGlobalCachingAllocator() {
    // Configure allocator with maximum cached bin size of ~1GB and no limit on
    // maximum cached bytes
    thread_local std::unique_ptr<cub::CachingDeviceAllocator> allocator{
        std::make_unique<cub::CachingDeviceAllocator>(2, 9, 29)};
    return *allocator;
  }
  pointer allocate(size_t n) {  // NOLINT
    pointer thrust_ptr;
    if (use_cub_allocator_) {
      T* raw_ptr{nullptr};
      auto errc =  GetGlobalCachingAllocator().DeviceAllocate(reinterpret_cast<void **>(&raw_ptr),
                                                              n * sizeof(T));
      if (errc != cudaSuccess) {
        ThrowOOMError("Caching allocator", n * sizeof(T));
      }
      thrust_ptr = pointer(raw_ptr);
    } else {
      try {
        thrust_ptr = SuperT::allocate(n);
        dh::safe_cuda(cudaGetLastError());
      } catch (const std::exception &e) {
        ThrowOOMError(e.what(), n * sizeof(T));
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
    : SuperT(rmm::cuda_stream_default, rmm::mr::get_current_device_resource()),
      use_cub_allocator_(!xgboost::GlobalConfigThreadLocalStore::Get()->use_rmm) {}
#endif  // defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1
  XGBOOST_DEVICE void construct(T *) {}  // NOLINT
 private:
  bool use_cub_allocator_{true};
};
}  // namespace detail

// Declare xgboost allocators
// Replacement of allocator with custom backend should occur here
template <typename T>
using XGBDeviceAllocator = detail::XGBDefaultDeviceAllocatorImpl<T>;
/*! Be careful that the initialization constructor is a no-op, which means calling
 *  `vec.resize(n)` won't initialize the memory region to 0. Instead use
 * `vec.resize(n, 0)`*/
template <typename T>
using XGBCachingDeviceAllocator = detail::XGBCachingDeviceAllocatorImpl<T>;
/** \brief Specialisation of thrust device vector using custom allocator. */
template <typename T>
using device_vector = thrust::device_vector<T,  XGBDeviceAllocator<T>>;  // NOLINT
template <typename T>
using caching_device_vector = thrust::device_vector<T,  XGBCachingDeviceAllocator<T>>;  // NOLINT

// Faster to instantiate than caching_device_vector and invokes no synchronisation
// Use this where vector functionality (e.g. resize) is not required
template <typename T>
class TemporaryArray {
 public:
  using AllocT = XGBCachingDeviceAllocator<T>;
  using value_type = T;  // NOLINT
  explicit TemporaryArray(size_t n) : size_(n) { ptr_ = AllocT().allocate(n); }
  TemporaryArray(size_t n, T val) : size_(n) {
    ptr_ = AllocT().allocate(n);
    this->fill(val);
  }
  ~TemporaryArray() { AllocT().deallocate(ptr_, this->size()); }
  void fill(T val)  // NOLINT
  {
    int device = 0;
    dh::safe_cuda(cudaGetDevice(&device));
    auto d_data = ptr_.get();
    LaunchN(this->size(), [=] __device__(size_t idx) { d_data[idx] = val; });
  }
  thrust::device_ptr<T> data() { return ptr_; }  // NOLINT
  size_t size() { return size_; }  // NOLINT

 private:
  thrust::device_ptr<T> ptr_;
  size_t size_;
};

/**
 * \brief A double buffer, useful for algorithms like sort.
 */
template <typename T>
class DoubleBuffer {
 public:
  cub::DoubleBuffer<T> buff;
  xgboost::common::Span<T> a, b;
  DoubleBuffer() = default;
  template <typename VectorT>
  DoubleBuffer(VectorT *v1, VectorT *v2) {
    a = xgboost::common::Span<T>(v1->data().get(), v1->size());
    b = xgboost::common::Span<T>(v2->data().get(), v2->size());
    buff = cub::DoubleBuffer<T>(a.data(), b.data());
  }

  size_t Size() const {
    CHECK_EQ(a.size(), b.size());
    return a.size();
  }
  cub::DoubleBuffer<T> &CubBuffer() { return buff; }

  T *Current() { return buff.Current(); }
  xgboost::common::Span<T> CurrentSpan() {
    return xgboost::common::Span<T>{buff.Current(), Size()};
  }

  T *Other() { return buff.Alternate(); }
};

template <typename T>
xgboost::common::Span<T> LazyResize(xgboost::Context const *ctx,
                                    xgboost::HostDeviceVector<T> *buffer, std::size_t n) {
  buffer->SetDevice(ctx->Device());
  if (buffer->Size() < n) {
    buffer->Resize(n);
  }
  return buffer->DeviceSpan().subspan(0, n);
}

/**
 * \brief Copies device span to std::vector.
 *
 * \tparam  T Generic type parameter.
 * \param [in,out]  dst Copy destination.
 * \param           src Copy source. Must be device memory.
 */
template <typename T>
void CopyDeviceSpanToVector(std::vector<T> *dst, xgboost::common::Span<T> src) {
  CHECK_EQ(dst->size(), src.size());
  dh::safe_cuda(cudaMemcpyAsync(dst->data(), src.data(), dst->size() * sizeof(T),
                                cudaMemcpyDeviceToHost));
}

/**
 * \brief Copies const device span to std::vector.
 *
 * \tparam  T Generic type parameter.
 * \param [in,out]  dst Copy destination.
 * \param           src Copy source. Must be device memory.
 */
template <typename T>
void CopyDeviceSpanToVector(std::vector<T> *dst, xgboost::common::Span<const T> src) {
  CHECK_EQ(dst->size(), src.size());
  dh::safe_cuda(cudaMemcpyAsync(dst->data(), src.data(), dst->size() * sizeof(T),
                                cudaMemcpyDeviceToHost));
}

template <class HContainer, class DContainer>
void CopyToD(HContainer const &h, DContainer *d) {
  if (h.empty()) {
    d->clear();
    return;
  }
  d->resize(h.size());
  using HVT = std::remove_cv_t<typename HContainer::value_type>;
  using DVT = std::remove_cv_t<typename DContainer::value_type>;
  static_assert(std::is_same<HVT, DVT>::value,
                "Host and device containers must have same value type.");
  dh::safe_cuda(cudaMemcpyAsync(d->data().get(), h.data(), h.size() * sizeof(HVT),
                                cudaMemcpyHostToDevice));
}

// Keep track of pinned memory allocation
struct PinnedMemory {
  void *temp_storage{nullptr};
  size_t temp_storage_bytes{0};

  ~PinnedMemory() { Free(); }

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

  template <typename T>
  xgboost::common::Span<T> GetSpan(size_t size, T init) {
    auto result = this->GetSpan<T>(size);
    for (auto &e : result) {
      e = init;
    }
    return result;
  }

  void Free() {
    if (temp_storage != nullptr) {
      safe_cuda(cudaFreeHost(temp_storage));
    }
  }
};

/*
 *  Utility functions
 */

/**
* @brief Helper function to perform device-wide sum-reduction, returns to the
* host
* @param in the input array to be reduced
* @param nVals number of elements in the input array
*/
template <typename T>
typename std::iterator_traits<T>::value_type SumReduction(T in, int nVals) {
  using ValueT = typename std::iterator_traits<T>::value_type;
  size_t tmpSize {0};
  ValueT *dummy_out = nullptr;
  dh::safe_cuda(cub::DeviceReduce::Sum(nullptr, tmpSize, in, dummy_out, nVals));

  TemporaryArray<char> temp(tmpSize + sizeof(ValueT));
  auto ptr = reinterpret_cast<ValueT *>(temp.data().get()) + 1;
  dh::safe_cuda(cub::DeviceReduce::Sum(
      reinterpret_cast<void *>(ptr), tmpSize, in,
      reinterpret_cast<ValueT *>(temp.data().get()),
      nVals));
  ValueT sum;
  dh::safe_cuda(cudaMemcpy(&sum, temp.data().get(), sizeof(ValueT),
                           cudaMemcpyDeviceToHost));
  return sum;
}

constexpr std::pair<int, int> CUDAVersion() {
#if defined(__CUDACC_VER_MAJOR__)
  return std::make_pair(__CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__);
#else
  // clang/clang-tidy
  return std::make_pair((CUDA_VERSION) / 1000, (CUDA_VERSION) % 100 / 10);
#endif  // defined(__CUDACC_VER_MAJOR__)
}

constexpr std::pair<int32_t, int32_t> ThrustVersion() {
  return std::make_pair(THRUST_MAJOR_VERSION, THRUST_MINOR_VERSION);
}
// Whether do we have thrust 1.x with x >= minor
template <int32_t minor>
constexpr bool HasThrustMinorVer() {
  return (ThrustVersion().first == 1 && ThrustVersion().second >= minor) ||
         ThrustVersion().first > 1;
}

namespace detail {
template <typename T>
using TypedDiscardCTK114 = thrust::discard_iterator<T>;

template <typename T>
class TypedDiscard : public thrust::discard_iterator<T> {
 public:
  using value_type = T;  // NOLINT
};
} // namespace detail

template <typename T>
using TypedDiscard =
    std::conditional_t<HasThrustMinorVer<12>(), detail::TypedDiscardCTK114<T>,
                       detail::TypedDiscard<T>>;

template <typename VectorT, typename T = typename VectorT::value_type,
  typename IndexT = typename xgboost::common::Span<T>::index_type>
xgboost::common::Span<T> ToSpan(
    VectorT &vec,
    IndexT offset = 0,
    IndexT size = std::numeric_limits<size_t>::max()) {
  size = size == std::numeric_limits<size_t>::max() ? vec.size() : size;
  CHECK_LE(offset + size, vec.size());
  return {vec.data().get() + offset, size};
}

template <typename T>
xgboost::common::Span<T> ToSpan(thrust::device_vector<T>& vec,
                                size_t offset, size_t size) {
  return ToSpan(vec, offset, size);
}

// thrust begin, similiar to std::begin
template <typename T>
thrust::device_ptr<T> tbegin(xgboost::HostDeviceVector<T>& vector) {  // NOLINT
  return thrust::device_ptr<T>(vector.DevicePointer());
}

template <typename T>
thrust::device_ptr<T> tend(xgboost::HostDeviceVector<T>& vector) {  // // NOLINT
  return tbegin(vector) + vector.Size();
}

template <typename T>
thrust::device_ptr<T const> tcbegin(xgboost::HostDeviceVector<T> const& vector) {  // NOLINT
  return thrust::device_ptr<T const>(vector.ConstDevicePointer());
}

template <typename T>
thrust::device_ptr<T const> tcend(xgboost::HostDeviceVector<T> const& vector) {  // NOLINT
  return tcbegin(vector) + vector.Size();
}

template <typename T>
XGBOOST_DEVICE thrust::device_ptr<T> tbegin(xgboost::common::Span<T>& span) {  // NOLINT
  return thrust::device_ptr<T>(span.data());
}

template <typename T>
XGBOOST_DEVICE thrust::device_ptr<T> tbegin(xgboost::common::Span<T> const& span) {  // NOLINT
  return thrust::device_ptr<T>(span.data());
}

template <typename T>
XGBOOST_DEVICE thrust::device_ptr<T> tend(xgboost::common::Span<T>& span) {  // NOLINT
  return tbegin(span) + span.size();
}

template <typename T>
XGBOOST_DEVICE thrust::device_ptr<T> tend(xgboost::common::Span<T> const& span) {  // NOLINT
  return tbegin(span) + span.size();
}

template <typename T>
XGBOOST_DEVICE auto trbegin(xgboost::common::Span<T> &span) {  // NOLINT
  return thrust::make_reverse_iterator(span.data() + span.size());
}

template <typename T>
XGBOOST_DEVICE auto trend(xgboost::common::Span<T> &span) {  // NOLINT
  return trbegin(span) + span.size();
}

template <typename T>
XGBOOST_DEVICE thrust::device_ptr<T const> tcbegin(xgboost::common::Span<T> const& span) {  // NOLINT
  return thrust::device_ptr<T const>(span.data());
}

template <typename T>
XGBOOST_DEVICE thrust::device_ptr<T const> tcend(xgboost::common::Span<T> const& span) {  // NOLINT
  return tcbegin(span) + span.size();
}

template <typename T>
XGBOOST_DEVICE auto tcrbegin(xgboost::common::Span<T> const &span) {  // NOLINT
  return thrust::make_reverse_iterator(span.data() + span.size());
}

template <typename T>
XGBOOST_DEVICE auto tcrend(xgboost::common::Span<T> const &span) {  // NOLINT
  return tcrbegin(span) + span.size();
}

// Atomic add function for gradients
template <typename OutputGradientT, typename InputGradientT>
XGBOOST_DEV_INLINE void AtomicAddGpair(OutputGradientT* dest,
                                       const InputGradientT& gpair) {
  auto dst_ptr = reinterpret_cast<typename OutputGradientT::ValueT*>(dest);

  atomicAdd(dst_ptr,
            static_cast<typename OutputGradientT::ValueT>(gpair.GetGrad()));
  atomicAdd(dst_ptr + 1,
            static_cast<typename OutputGradientT::ValueT>(gpair.GetHess()));
}


// Thrust version of this function causes error on Windows
template <typename ReturnT, typename IterT, typename FuncT>
XGBOOST_DEVICE thrust::transform_iterator<FuncT, IterT, ReturnT> MakeTransformIterator(
  IterT iter, FuncT func) {
  return thrust::transform_iterator<FuncT, IterT, ReturnT>(iter, func);
}

template <typename It>
size_t XGBOOST_DEVICE SegmentId(It first, It last, size_t idx) {
  size_t segment_id = thrust::upper_bound(thrust::seq, first, last, idx) - 1 - first;
  return segment_id;
}

template <typename T>
size_t XGBOOST_DEVICE SegmentId(xgboost::common::Span<T> segments_ptr, size_t idx) {
  return SegmentId(segments_ptr.cbegin(), segments_ptr.cend(), idx);
}

namespace detail {
template <typename Key, typename KeyOutIt>
struct SegmentedUniqueReduceOp {
  KeyOutIt key_out;
  __device__ Key const& operator()(Key const& key) const {
    auto constexpr kOne = static_cast<std::remove_reference_t<decltype(*(key_out + key.first))>>(1);
    atomicAdd(&(*(key_out + key.first)), kOne);
    return key;
  }
};
}  // namespace detail

/* \brief Segmented unique function.  Keys are pointers to segments with key_segments_last -
 *        key_segments_first = n_segments + 1.
 *
 * \pre   Input segment and output segment must not overlap.
 *
 * \param key_segments_first Beginning iterator of segments.
 * \param key_segments_last  End iterator of segments.
 * \param val_first          Beginning iterator of values.
 * \param val_last           End iterator of values.
 * \param key_segments_out   Output iterator of segments.
 * \param val_out            Output iterator of values.
 *
 * \return Number of unique values in total.
 */
template <typename DerivedPolicy, typename KeyInIt, typename KeyOutIt, typename ValInIt,
          typename ValOutIt, typename CompValue, typename CompKey>
size_t
SegmentedUnique(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                KeyInIt key_segments_first, KeyInIt key_segments_last, ValInIt val_first,
                ValInIt val_last, KeyOutIt key_segments_out, ValOutIt val_out,
                CompValue comp, CompKey comp_key=thrust::equal_to<size_t>{}) {
  using Key = thrust::pair<size_t, typename thrust::iterator_traits<ValInIt>::value_type>;
  auto unique_key_it = dh::MakeTransformIterator<Key>(
      thrust::make_counting_iterator(static_cast<size_t>(0)),
      [=] __device__(size_t i) {
        size_t seg = dh::SegmentId(key_segments_first, key_segments_last, i);
        return thrust::make_pair(seg, *(val_first + i));
      });
  size_t segments_len = key_segments_last - key_segments_first;
  thrust::fill(thrust::device, key_segments_out, key_segments_out + segments_len, 0);
  size_t n_inputs = std::distance(val_first, val_last);
  // Reduce the number of uniques elements per segment, avoid creating an intermediate
  // array for `reduce_by_key`.  It's limited by the types that atomicAdd supports.  For
  // example, size_t is not supported as of CUDA 10.2.
  auto reduce_it = thrust::make_transform_output_iterator(
      thrust::make_discard_iterator(),
      detail::SegmentedUniqueReduceOp<Key, KeyOutIt>{key_segments_out});
  auto uniques_ret = thrust::unique_by_key_copy(
      exec, unique_key_it, unique_key_it + n_inputs,
      val_first, reduce_it, val_out,
      [=] __device__(Key const &l, Key const &r) {
        if (comp_key(l.first, r.first)) {
          // In the same segment.
          return comp(l.second, r.second);
        }
        return false;
      });
  auto n_uniques = uniques_ret.second - val_out;
  CHECK_LE(n_uniques, n_inputs);
  thrust::exclusive_scan(exec, key_segments_out,
                         key_segments_out + segments_len, key_segments_out, 0);
  return n_uniques;
}

template <typename... Inputs,
          std::enable_if_t<std::tuple_size<std::tuple<Inputs...>>::value == 7>
              * = nullptr>
size_t SegmentedUnique(Inputs &&...inputs) {
  dh::XGBCachingDeviceAllocator<char> alloc;
  return SegmentedUnique(thrust::cuda::par(alloc),
                         std::forward<Inputs &&>(inputs)...,
                         thrust::equal_to<size_t>{});
}

/**
 * \brief Unique by key for many groups of data.  Has same constraint as `SegmentedUnique`.
 *
 * \tparam exec               thrust execution policy
 * \tparam key_segments_first start iter to segment pointer
 * \tparam key_segments_last  end iter to segment pointer
 * \tparam key_first          start iter to key for comparison
 * \tparam key_last           end iter to key for comparison
 * \tparam val_first          start iter to values
 * \tparam key_segments_out   output iterator for new segment pointer
 * \tparam val_out            output iterator for values
 * \tparam comp               binary comparison operator
 */
template <typename DerivedPolicy, typename SegInIt, typename SegOutIt,
          typename KeyInIt, typename ValInIt, typename ValOutIt, typename Comp>
size_t SegmentedUniqueByKey(
    const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
    SegInIt key_segments_first, SegInIt key_segments_last, KeyInIt key_first,
    KeyInIt key_last, ValInIt val_first, SegOutIt key_segments_out,
    ValOutIt val_out, Comp comp) {
  using Key =
      thrust::pair<size_t,
                   typename thrust::iterator_traits<KeyInIt>::value_type>;

  auto unique_key_it = dh::MakeTransformIterator<Key>(
      thrust::make_counting_iterator(static_cast<size_t>(0)),
      [=] __device__(size_t i) {
        size_t seg = dh::SegmentId(key_segments_first, key_segments_last, i);
        return thrust::make_pair(seg, *(key_first + i));
      });
  size_t segments_len = key_segments_last - key_segments_first;
  thrust::fill(thrust::device, key_segments_out,
               key_segments_out + segments_len, 0);
  size_t n_inputs = std::distance(key_first, key_last);
  // Reduce the number of uniques elements per segment, avoid creating an
  // intermediate array for `reduce_by_key`.  It's limited by the types that
  // atomicAdd supports.  For example, size_t is not supported as of CUDA 10.2.
  auto reduce_it = thrust::make_transform_output_iterator(
      thrust::make_discard_iterator(),
      detail::SegmentedUniqueReduceOp<Key, SegOutIt>{key_segments_out});
  auto uniques_ret = thrust::unique_by_key_copy(
      exec, unique_key_it, unique_key_it + n_inputs, val_first, reduce_it,
      val_out, [=] __device__(Key const &l, Key const &r) {
        if (l.first == r.first) {
          // In the same segment.
          return comp(thrust::get<1>(l), thrust::get<1>(r));
        }
        return false;
      });
  auto n_uniques = uniques_ret.second - val_out;
  CHECK_LE(n_uniques, n_inputs);
  thrust::exclusive_scan(exec, key_segments_out,
                         key_segments_out + segments_len, key_segments_out, 0);
  return n_uniques;
}

template <typename Policy, typename InputIt, typename Init, typename Func>
auto Reduce(Policy policy, InputIt first, InputIt second, Init init, Func reduce_op) {
  size_t constexpr kLimit = std::numeric_limits<int32_t>::max() / 2;
  size_t size = std::distance(first, second);
  using Ty = std::remove_cv_t<Init>;
  Ty aggregate = init;
  for (size_t offset = 0; offset < size; offset += kLimit) {
    auto begin_it = first + offset;
    auto end_it = first + std::min(offset + kLimit, size);
    size_t batch_size = std::distance(begin_it, end_it);
    CHECK_LE(batch_size, size);
    auto ret = thrust::reduce(policy, begin_it, end_it, init, reduce_op);
    aggregate = reduce_op(aggregate, ret);
  }
  return aggregate;
}

// wrapper to avoid integer `num_items`.
template <typename InputIteratorT, typename OutputIteratorT, typename ScanOpT,
          typename OffsetT>
void InclusiveScan(InputIteratorT d_in, OutputIteratorT d_out, ScanOpT scan_op,
                   OffsetT num_items) {
  size_t bytes = 0;
#if THRUST_MAJOR_VERSION >= 2
  safe_cuda((
      cub::DispatchScan<InputIteratorT, OutputIteratorT, ScanOpT, cub::NullType,
                        OffsetT>::Dispatch(nullptr, bytes, d_in, d_out, scan_op,
                                           cub::NullType(), num_items, nullptr)));
#else
  safe_cuda((
      cub::DispatchScan<InputIteratorT, OutputIteratorT, ScanOpT, cub::NullType,
                        OffsetT>::Dispatch(nullptr, bytes, d_in, d_out, scan_op,
                                           cub::NullType(), num_items, nullptr,
                                           false)));
#endif
  TemporaryArray<char> storage(bytes);
#if THRUST_MAJOR_VERSION >= 2
  safe_cuda((
      cub::DispatchScan<InputIteratorT, OutputIteratorT, ScanOpT, cub::NullType,
                        OffsetT>::Dispatch(storage.data().get(), bytes, d_in,
                                           d_out, scan_op, cub::NullType(),
                                           num_items, nullptr)));
#else
  safe_cuda((
      cub::DispatchScan<InputIteratorT, OutputIteratorT, ScanOpT, cub::NullType,
                        OffsetT>::Dispatch(storage.data().get(), bytes, d_in,
                                           d_out, scan_op, cub::NullType(),
                                           num_items, nullptr, false)));
#endif
}

template <typename InIt, typename OutIt, typename Predicate>
void CopyIf(InIt in_first, InIt in_second, OutIt out_first, Predicate pred) {
  // We loop over batches because thrust::copy_if can't deal with sizes > 2^31
  // See thrust issue #1302, XGBoost #6822
  size_t constexpr kMaxCopySize = std::numeric_limits<int>::max() / 2;
  size_t length = std::distance(in_first, in_second);
  XGBCachingDeviceAllocator<char> alloc;
  for (size_t offset = 0; offset < length; offset += kMaxCopySize) {
    auto begin_input = in_first + offset;
    auto end_input = in_first + std::min(offset + kMaxCopySize, length);
    out_first = thrust::copy_if(thrust::cuda::par(alloc), begin_input,
                                end_input, out_first, pred);
  }
}

template <typename InputIteratorT, typename OutputIteratorT, typename OffsetT>
void InclusiveSum(InputIteratorT d_in, OutputIteratorT d_out, OffsetT num_items) {
  InclusiveScan(d_in, d_out, cub::Sum(), num_items);
}

class CUDAStreamView;

class CUDAEvent {
  cudaEvent_t event_{nullptr};

 public:
  CUDAEvent() { dh::safe_cuda(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming)); }
  ~CUDAEvent() {
    if (event_) {
      dh::safe_cuda(cudaEventDestroy(event_));
    }
  }

  CUDAEvent(CUDAEvent const &that) = delete;
  CUDAEvent &operator=(CUDAEvent const &that) = delete;

  inline void Record(CUDAStreamView stream);       // NOLINT

  operator cudaEvent_t() const { return event_; }  // NOLINT
};

class CUDAStreamView {
  cudaStream_t stream_{nullptr};

 public:
  explicit CUDAStreamView(cudaStream_t s) : stream_{s} {}
  void Wait(CUDAEvent const &e) {
#if defined(__CUDACC_VER_MAJOR__)
#if __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ == 0
    // CUDA == 11.0
    dh::safe_cuda(cudaStreamWaitEvent(stream_, cudaEvent_t{e}, 0));
#else
    // CUDA > 11.0
    dh::safe_cuda(cudaStreamWaitEvent(stream_, cudaEvent_t{e}, cudaEventWaitDefault));
#endif  // __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ == 0:
#else   // clang
    dh::safe_cuda(cudaStreamWaitEvent(stream_, cudaEvent_t{e}, cudaEventWaitDefault));
#endif  //  defined(__CUDACC_VER_MAJOR__)
  }
  operator cudaStream_t() const {  // NOLINT
    return stream_;
  }
  cudaError_t Sync(bool error = true) {
    if (error) {
      dh::safe_cuda(cudaStreamSynchronize(stream_));
      return cudaSuccess;
    }
    return cudaStreamSynchronize(stream_);
  }
};

inline void CUDAEvent::Record(CUDAStreamView stream) {  // NOLINT
  dh::safe_cuda(cudaEventRecord(event_, cudaStream_t{stream}));
}

// Changing this has effect on prediction return, where we need to pass the pointer to
// third-party libraries like cuPy
inline CUDAStreamView DefaultStream() {
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
  return CUDAStreamView{cudaStreamPerThread};
#else
  return CUDAStreamView{cudaStreamLegacy};
#endif
}

class CUDAStream {
  cudaStream_t stream_;

 public:
  CUDAStream() { dh::safe_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
  ~CUDAStream() { dh::safe_cuda(cudaStreamDestroy(stream_)); }

  [[nodiscard]] CUDAStreamView View() const { return CUDAStreamView{stream_}; }
  [[nodiscard]] cudaStream_t Handle() const { return stream_; }

  void Sync() { this->View().Sync(); }
};

// Force nvcc to load data as constant
template <typename T>
class LDGIterator {
  using DeviceWordT = typename cub::UnitWord<T>::DeviceWord;
  static constexpr std::size_t kNumWords = sizeof(T) / sizeof(DeviceWordT);

  const T *ptr_;

 public:
  XGBOOST_DEVICE explicit LDGIterator(const T *ptr) : ptr_(ptr) {}
  __device__ T operator[](std::size_t idx) const {
    DeviceWordT tmp[kNumWords];
    static_assert(sizeof(tmp) == sizeof(T), "Expect sizes to be equal.");
#pragma unroll
    for (int i = 0; i < kNumWords; i++) {
      tmp[i] = __ldg(reinterpret_cast<const DeviceWordT *>(ptr_ + idx) + i);
    }
    return *reinterpret_cast<const T *>(tmp);
  }
};
}  // namespace dh
