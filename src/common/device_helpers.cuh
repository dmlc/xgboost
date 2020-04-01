/*!
 * Copyright 2017-2020 XGBoost contributors
 */
#pragma once
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
#include <thrust/logical.h>
#include <thrust/gather.h>

#include <omp.h>
#include <rabit/rabit.h>
#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "xgboost/logging.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/span.h"

#include "common.h"
#include "timer.h"

#ifdef XGBOOST_USE_NCCL
#include "nccl.h"
#include "../common/io.h"
#endif

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

#define HOST_DEV_INLINE XGBOOST_DEVICE __forceinline__
#define DEV_INLINE __device__ __forceinline__

#ifdef XGBOOST_USE_NCCL
#define safe_nccl(ans) ThrowOnNcclError((ans), __FILE__, __LINE__)

inline ncclResult_t ThrowOnNcclError(ncclResult_t code, const char *file,
                                        int line) {
  if (code != ncclSuccess) {
    std::stringstream ss;
    ss << "NCCL failure :" << ncclGetErrorString(code) << " ";
    ss << file << "(" << line << ")";
    throw std::runtime_error(ss.str());
  }

  return code;
}
#endif

inline int32_t CudaGetPointerDevice(void* ptr) {
  int32_t device = -1;
  cudaPointerAttributes attr;
  dh::safe_cuda(cudaPointerGetAttributes(&attr, ptr));
  device = attr.device;
  return device;
}

inline void CudaCheckPointerDevice(void* ptr) {
  auto ptr_device = CudaGetPointerDevice(ptr);
  int cur_device = -1;
  dh::safe_cuda(cudaGetDevice(&cur_device));
  CHECK_EQ(ptr_device, cur_device) << "pointer device: " << ptr_device
                                   << "current device: " << cur_device;
}

template <typename T>
const T *Raw(const thrust::device_vector<T> &v) {  //  NOLINT
  return raw_pointer_cast(v.data());
}

inline size_t AvailableMemory(int device_idx) {
  size_t device_free = 0;
  size_t device_total = 0;
  safe_cuda(cudaSetDevice(device_idx));
  dh::safe_cuda(cudaMemGetInfo(&device_free, &device_total));
  return device_free;
}

inline size_t TotalMemory(int device_idx) {
  size_t device_free = 0;
  size_t device_total = 0;
  safe_cuda(cudaSetDevice(device_idx));
  dh::safe_cuda(cudaMemGetInfo(&device_free, &device_total));
  return device_total;
}

/**
 * \fn  inline int max_shared_memory(int device_idx)
 *
 * \brief Maximum shared memory per block on this device.
 *
 * \param device_idx  Zero-based index of the device.
 */

inline size_t MaxSharedMemory(int device_idx) {
  cudaDeviceProp prop;
  dh::safe_cuda(cudaGetDeviceProperties(&prop, device_idx));
  return prop.sharedMemPerBlock;
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

DEV_INLINE void AtomicOrByte(unsigned int* __restrict__ buffer, size_t ibyte, unsigned char b) {
  atomicOr(&buffer[ibyte / sizeof(unsigned int)],
           static_cast<unsigned int>(b) << (ibyte % (sizeof(unsigned int)) * 8));
}

namespace internal {

// Items of size 'n' are sorted in an order determined by the Comparator
// If left is true, find the number of elements where 'comp(item, v)' returns true;
// 0 if nothing is true
// If left is false, find the number of elements where '!comp(item, v)' returns true;
// 0 if nothing is true
template <typename T, typename Comparator = thrust::greater<T>>
XGBOOST_DEVICE __forceinline__ uint32_t
CountNumItemsImpl(bool left, const T * __restrict__ items, uint32_t n, T v,
                  const Comparator &comp = Comparator()) {
  const T *items_begin = items;
  uint32_t num_remaining = n;
  const T *middle_item = nullptr;
  uint32_t middle;
  while (num_remaining > 0) {
    middle_item = items_begin;
    middle = num_remaining / 2;
    middle_item += middle;
    if ((left && comp(*middle_item, v)) || (!left && !comp(v, *middle_item))) {
      items_begin = ++middle_item;
      num_remaining -= middle + 1;
    } else {
      num_remaining = middle;
    }
  }

  return left ? items_begin - items : items + n - items_begin;
}

}  // namespace internal

/*!
 * \brief Find the strict upper bound for an element in a sorted array
 *        using binary search.
 * \param items pointer to the first element of the sorted array
 * \param n length of the sorted array
 * \param v value for which to find the upper bound
 * \param comp determines how the items are sorted ascending/descending order - should conform
 *        to ordering semantics
 * \return the smallest index i that has a value > v, or n if none is larger when sorted ascendingly
 *         or, an index i with a value < v, or 0 if none is smaller when sorted descendingly
*/
// Preserve existing default behavior of upper bound
template <typename T, typename Comp = thrust::less<T>>
XGBOOST_DEVICE __forceinline__ uint32_t UpperBound(const T *__restrict__ items,
                                                   uint32_t n,
                                                   T v,
                                                   const Comp &comp = Comp()) {
  if (std::is_same<Comp, thrust::less<T>>::value ||
      std::is_same<Comp, thrust::greater<T>>::value) {
    return n - internal::CountNumItemsImpl(false, items, n, v, comp);
  } else {
    static_assert(std::is_same<Comp, thrust::less<T>>::value ||
                  std::is_same<Comp, thrust::greater<T>>::value,
                  "Invalid comparator used in Upperbound - can only be thrust::greater/less");
    return std::numeric_limits<uint32_t>::max(); // Simply to quiesce the compiler
  }
}

/*!
 * \brief Find the strict lower bound for an element in a sorted array
 *        using binary search.
 * \param items pointer to the first element of the sorted array
 * \param n length of the sorted array
 * \param v value for which to find the upper bound
 * \param comp determines how the items are sorted ascending/descending order - should conform
 *        to ordering semantics
 * \return the smallest index i that has a value >= v, or n if none is larger
 *         when sorted ascendingly
 *         or, an index i with a value <= v, or 0 if none is smaller when sorted descendingly
*/
template <typename T, typename Comp = thrust::less<T>>
XGBOOST_DEVICE __forceinline__ uint32_t LowerBound(const T *__restrict__ items,
                                                   uint32_t n,
                                                   T v,
                                                   const Comp &comp = Comp()) {
  if (std::is_same<Comp, thrust::less<T>>::value ||
      std::is_same<Comp, thrust::greater<T>>::value) {
    return internal::CountNumItemsImpl(true, items, n, v, comp);
  } else {
    static_assert(std::is_same<Comp, thrust::less<T>>::value ||
                  std::is_same<Comp, thrust::greater<T>>::value,
                  "Invalid comparator used in LowerBound - can only be thrust::greater/less");
    return std::numeric_limits<uint32_t>::max(); // Simply to quiesce the compiler
  }
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
template <typename L>
__global__ void LaunchNKernel(int device_idx, size_t begin, size_t end,
                              L lambda) {
  for (auto i : GridStrideRange(begin, end)) {
    lambda(i, device_idx);
  }
}

/* \brief A wrapper around kernel launching syntax, used to guard against empty input.
 *
 * - nvcc fails to deduce template argument when kernel is a template accepting __device__
 *   function as argument.  Hence functions like `LaunchN` cannot use this wrapper.
 *
 * - With c++ initialization list `{}` syntax, you are forced to comply with the CUDA type
 *   spcification.
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
inline void LaunchN(int device_idx, size_t n, cudaStream_t stream, L lambda) {
  if (n == 0) {
    return;
  }
  safe_cuda(cudaSetDevice(device_idx));
  const int GRID_SIZE =
      static_cast<int>(xgboost::common::DivRoundUp(n, ITEMS_PER_THREAD * BLOCK_THREADS));
  LaunchNKernel<<<GRID_SIZE, BLOCK_THREADS, 0, stream>>>(  // NOLINT
      static_cast<size_t>(0), n, lambda);
}

// Default stream version
template <int ITEMS_PER_THREAD = 8, int BLOCK_THREADS = 256, typename L>
inline void LaunchN(int device_idx, size_t n, L lambda) {
  LaunchN<ITEMS_PER_THREAD, BLOCK_THREADS>(device_idx, n, nullptr, lambda);
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
      peak_allocated_bytes =
        std::max(peak_allocated_bytes, currently_allocated_bytes);
      num_allocations++;
      CHECK_GT(num_allocations, num_deallocations);
    }
    void RegisterDeallocation(void *ptr, size_t n, int current_device) {
      auto itr = device_allocations.find(ptr);
      if (itr == device_allocations.end()) {
        LOG(FATAL) << "Attempting to deallocate " << n << " bytes on device "
                   << current_device << " that was never allocated ";
      }
      num_deallocations++;
      CHECK_LE(num_deallocations, num_allocations);
      currently_allocated_bytes -= itr->second;
      device_allocations.erase(itr);
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
  size_t PeakMemory()
  {
    return stats_.peak_allocated_bytes;
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
    auto rank = rabit::GetRank();
    LOG(DEBUG) << "R:" << rank << ": " << file << ":" << line;
  }
  safe_cuda(cudaDeviceSynchronize());
  safe_cuda(cudaGetLastError());
}

namespace detail {
/**
 * \brief Default memory allocator, uses cudaMalloc/Free and logs allocations if verbose.
 */
template <class T>
struct XGBDefaultDeviceAllocatorImpl : thrust::device_malloc_allocator<T> {
  using SuperT = thrust::device_malloc_allocator<T>;
  using pointer = thrust::device_ptr<T>;  // NOLINT
  template<typename U>
  struct rebind  // NOLINT
  {
    using other = XGBDefaultDeviceAllocatorImpl<U>;  // NOLINT
  };
  pointer allocate(size_t n) {  // NOLINT
    pointer ptr = SuperT::allocate(n);
    GlobalMemoryLogger().RegisterAllocation(ptr.get(), n * sizeof(T));
    return ptr;
  }
  void deallocate(pointer ptr, size_t n) {  // NOLINT
    GlobalMemoryLogger().RegisterDeallocation(ptr.get(), n * sizeof(T));
    return SuperT::deallocate(ptr, n);
  }
};

/**
 * \brief Caching memory allocator, uses cub::CachingDeviceAllocator as a back-end and logs allocations if verbose. Does not initialise memory on construction.
 */
template <class T>
struct XGBCachingDeviceAllocatorImpl : thrust::device_malloc_allocator<T> {
  using pointer = thrust::device_ptr<T>;  // NOLINT
  template<typename U>
  struct rebind  // NOLINT
  {
    using other = XGBCachingDeviceAllocatorImpl<U>;  // NOLINT
  };
   cub::CachingDeviceAllocator& GetGlobalCachingAllocator ()
   {
     // Configure allocator with maximum cached bin size of ~1GB and no limit on
     // maximum cached bytes
     static cub::CachingDeviceAllocator *allocator = new cub::CachingDeviceAllocator(2, 9, 29);
     return *allocator;
   }
   pointer allocate(size_t n) {  // NOLINT
     T *ptr;
     GetGlobalCachingAllocator().DeviceAllocate(reinterpret_cast<void **>(&ptr),
                                                n * sizeof(T));
     pointer thrust_ptr(ptr);
     GlobalMemoryLogger().RegisterAllocation(thrust_ptr.get(), n * sizeof(T));
     return thrust_ptr;
   }
   void deallocate(pointer ptr, size_t n) {  // NOLINT
     GlobalMemoryLogger().RegisterDeallocation(ptr.get(), n * sizeof(T));
     GetGlobalCachingAllocator().DeviceFree(ptr.get());
   }
  __host__ __device__
    void construct(T *)  // NOLINT
  {
    // no-op
  }
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

/**
 * \brief Copies std::vector to device span.
 *
 * \tparam  T Generic type parameter.
 * \param dst Copy destination. Must be device memory.
 * \param src Copy source.
 */
template <typename T>
void CopyVectorToDeviceSpan(xgboost::common::Span<T> dst ,const std::vector<T>&src)
{
  CHECK_EQ(dst.size(), src.size());
  dh::safe_cuda(cudaMemcpyAsync(dst.data(), src.data(), dst.size() * sizeof(T),
                                cudaMemcpyHostToDevice));
}

/**
 * \brief Device to device memory copy from src to dst. Spans must be the same size. Use subspan to
 *        copy from a smaller array to a larger array.
 *
 * \tparam  T Generic type parameter.
 * \param dst Copy destination. Must be device memory.
 * \param src Copy source. Must be device memory.
 */
template <typename T>
void CopyDeviceSpan(xgboost::common::Span<T> dst,
                    xgboost::common::Span<T> src) {
  CHECK_EQ(dst.size(), src.size());
  dh::safe_cuda(cudaMemcpyAsync(dst.data(), src.data(), dst.size() * sizeof(T),
                                cudaMemcpyDeviceToDevice));
}

/*! \brief Helper for allocating large block of memory. */
class BulkAllocator {
  std::vector<char *> d_ptr_;
  std::vector<size_t> size_;
  int device_idx_{-1};

  static const int kAlign = 256;

  size_t AlignRoundUp(size_t n) const {
    n = (n + kAlign - 1) / kAlign;
    return n * kAlign;
  }

  template <typename T>
  size_t GetSizeBytes(xgboost::common::Span<T> *first_vec, size_t first_size) {
    return AlignRoundUp(first_size * sizeof(T));
  }

  template <typename T, typename... Args>
  size_t GetSizeBytes(xgboost::common::Span<T> *first_vec, size_t first_size, Args... args) {
    return GetSizeBytes<T>(first_vec, first_size) + GetSizeBytes(args...);
  }

  template <typename T>
  void AllocateSpan(int device_idx, char *ptr, xgboost::common::Span<T> *first_vec,
    size_t first_size) {
    *first_vec = xgboost::common::Span<T>(reinterpret_cast<T *>(ptr), first_size);
  }

  template <typename T, typename... Args>
  void AllocateSpan(int device_idx, char *ptr, xgboost::common::Span<T> *first_vec,
    size_t first_size, Args... args) {
    AllocateSpan<T>(device_idx, ptr, first_vec, first_size);
    ptr += AlignRoundUp(first_size * sizeof(T));
    AllocateSpan(device_idx, ptr, args...);
  }

  char *AllocateDevice(int device_idx, size_t bytes) {
    safe_cuda(cudaSetDevice(device_idx));
    XGBDeviceAllocator<char> allocator;
    return allocator.allocate(bytes).get();
  }

  template <typename T>
  size_t GetSizeBytes(DoubleBuffer<T> *first_vec, size_t first_size) {
    return 2 * AlignRoundUp(first_size * sizeof(T));
  }

  template <typename T, typename... Args>
  size_t GetSizeBytes(DoubleBuffer<T> *first_vec, size_t first_size, Args... args) {
    return GetSizeBytes<T>(first_vec, first_size) + GetSizeBytes(args...);
  }

  template <typename T>
  void AllocateSpan(int device_idx, char *ptr, DoubleBuffer<T> *first_vec,
                    size_t first_size) {
    auto ptr1 = reinterpret_cast<T *>(ptr);
    auto ptr2 = ptr1 + first_size;
    first_vec->a = xgboost::common::Span<T>(ptr1, first_size);
    first_vec->b = xgboost::common::Span<T>(ptr2, first_size);
    first_vec->buff.d_buffers[0] = ptr1;
    first_vec->buff.d_buffers[1] = ptr2;
    first_vec->buff.selector = 0;
  }

  template <typename T, typename... Args>
  void AllocateSpan(int device_idx, char *ptr, DoubleBuffer<T> *first_vec,
                     size_t first_size, Args... args) {
    AllocateSpan<T>(device_idx, ptr, first_vec, first_size);
    ptr += (AlignRoundUp(first_size * sizeof(T)) * 2);
    AllocateSpan(device_idx, ptr, args...);
  }

 public:
  BulkAllocator() = default;
  // prevent accidental copying, moving or assignment of this object
  BulkAllocator(const BulkAllocator&) = delete;
  BulkAllocator(BulkAllocator&&) = delete;
  void operator=(const BulkAllocator&) = delete;
  void operator=(BulkAllocator&&) = delete;

  /*!
   * \brief Clear the bulk allocator.
   *
   * This frees the GPU memory managed by this allocator.
   */
  void Clear() {
    if (d_ptr_.empty()) return;

    safe_cuda(cudaSetDevice(device_idx_));
    size_t idx = 0;
    std::for_each(d_ptr_.begin(), d_ptr_.end(), [&](char *dptr) {
      XGBDeviceAllocator<char>().deallocate(thrust::device_ptr<char>(dptr), size_[idx++]);
    });
    d_ptr_.clear();
    size_.clear();
  }

  ~BulkAllocator() {
    Clear();
  }

  // returns sum of bytes for all allocations
  size_t Size() {
    return std::accumulate(size_.begin(), size_.end(), static_cast<size_t>(0));
  }

  template <typename... Args>
  void Allocate(int device_idx, Args... args) {
    if (device_idx_ == -1) {
      device_idx_ = device_idx;
    }
    else CHECK(device_idx_ == device_idx);
    size_t size = GetSizeBytes(args...);

    char *ptr = AllocateDevice(device_idx, size);

    AllocateSpan(device_idx, ptr, args...);

    d_ptr_.push_back(ptr);
    size_.push_back(size);
  }
};

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

  void Free() {
    if (temp_storage != nullptr) {
      safe_cuda(cudaFreeHost(temp_storage));
    }
  }
};

// Keep track of cub library device allocation
struct CubMemory {
  void *d_temp_storage { nullptr };
  size_t temp_storage_bytes { 0 };

  // Thrust
  using value_type = char;  // NOLINT

  CubMemory() = default;

  ~CubMemory() { Free(); }

  template <typename T>
  xgboost::common::Span<T> GetSpan(size_t size) {
    this->LazyAllocate(size * sizeof(T));
    return xgboost::common::Span<T>(static_cast<T*>(d_temp_storage), size);
  }

  void Free() {
    if (this->IsAllocated()) {
      XGBDeviceAllocator<uint8_t> allocator;
      allocator.deallocate(thrust::device_ptr<uint8_t>(static_cast<uint8_t *>(d_temp_storage)),
                           temp_storage_bytes);
      d_temp_storage = nullptr;
      temp_storage_bytes = 0;
    }
  }

  void LazyAllocate(size_t num_bytes) {
    if (num_bytes > temp_storage_bytes) {
      Free();
      XGBDeviceAllocator<uint8_t> allocator;
      d_temp_storage = static_cast<void *>(allocator.allocate(num_bytes).get());
      temp_storage_bytes = num_bytes;
    }
  }
  // Thrust
  char *allocate(std::ptrdiff_t num_bytes) {  // NOLINT
    LazyAllocate(num_bytes);
    return reinterpret_cast<char *>(d_temp_storage);
  }

  // Thrust
  void deallocate(char *ptr, size_t n) {  // NOLINT

    // Do nothing
  }

  bool IsAllocated() { return d_temp_storage != nullptr; }
};

/*
 *  Utility functions
 */

// Load balancing search

template <typename CoordinateT, typename SegmentT, typename OffsetT>
void FindMergePartitions(int device_idx, CoordinateT *d_tile_coordinates,
                         size_t num_tiles, int tile_size, SegmentT segments,
                         OffsetT num_rows, OffsetT num_elements) {
  dh::LaunchN(device_idx, num_tiles + 1, [=] __device__(int idx) {
    OffsetT diagonal = idx * tile_size;
    CoordinateT tile_coordinate;
    cub::CountingInputIterator<OffsetT> nonzero_indices(0);

    // Search the merge path
    // Cast to signed integer as this function can have negatives
    cub::MergePathSearch(static_cast<int64_t>(diagonal), segments + 1,
                         nonzero_indices, static_cast<int64_t>(num_rows),
                         static_cast<int64_t>(num_elements), tile_coordinate);

    // Output starting offset
    d_tile_coordinates[idx] = tile_coordinate;
  });
}

template <int TILE_SIZE, int ITEMS_PER_THREAD, int BLOCK_THREADS,
          typename OffsetT, typename CoordinateT, typename FunctionT,
          typename SegmentIterT>
__global__ void LbsKernel(CoordinateT *d_coordinates,
                          SegmentIterT segment_end_offsets, FunctionT f,
                          OffsetT num_segments) {
  int tile = blockIdx.x;
  CoordinateT tile_start_coord = d_coordinates[tile];
  CoordinateT tile_end_coord = d_coordinates[tile + 1];
  int64_t tile_num_rows = tile_end_coord.x - tile_start_coord.x;
  int64_t tile_num_elements = tile_end_coord.y - tile_start_coord.y;

  cub::CountingInputIterator<OffsetT> tile_element_indices(tile_start_coord.y);
  CoordinateT thread_start_coord;

  using SegmentT = typename std::iterator_traits<SegmentIterT>::value_type;
  __shared__ struct {
    SegmentT tile_segment_end_offsets[TILE_SIZE + 1];
    SegmentT output_segment[TILE_SIZE];
  } temp_storage;

  for (auto item : dh::BlockStrideRange(int(0), int(tile_num_rows + 1))) {
    temp_storage.tile_segment_end_offsets[item] =
        segment_end_offsets[min(static_cast<size_t>(tile_start_coord.x + item),
                                static_cast<size_t>(num_segments - 1))];
  }
  __syncthreads();

  int64_t diag = threadIdx.x * ITEMS_PER_THREAD;

  // Cast to signed integer as this function can have negatives
  cub::MergePathSearch(diag,                                   // Diagonal
                       temp_storage.tile_segment_end_offsets,  // List A
                       tile_element_indices,                   // List B
                       tile_num_rows, tile_num_elements, thread_start_coord);

  CoordinateT thread_current_coord = thread_start_coord;
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (tile_element_indices[thread_current_coord.y] <
        temp_storage.tile_segment_end_offsets[thread_current_coord.x]) {
      temp_storage.output_segment[thread_current_coord.y] =
          thread_current_coord.x + tile_start_coord.x;
      ++thread_current_coord.y;
    } else {
      ++thread_current_coord.x;
    }
  }
  __syncthreads();

  for (auto item : dh::BlockStrideRange(int(0), int(tile_num_elements))) {
    f(tile_start_coord.y + item, temp_storage.output_segment[item]);
  }
}

template <typename FunctionT, typename SegmentIterT, typename OffsetT>
void SparseTransformLbs(int device_idx, dh::CubMemory *temp_memory,
                        OffsetT count, SegmentIterT segments,
                        OffsetT num_segments, FunctionT f) {
  typedef typename cub::CubVector<OffsetT, 2>::Type CoordinateT;
  dh::safe_cuda(cudaSetDevice(device_idx));
  const int BLOCK_THREADS = 256;
  const int ITEMS_PER_THREAD = 1;
  const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
  auto num_tiles = xgboost::common::DivRoundUp(count + num_segments, BLOCK_THREADS);
  CHECK(num_tiles < std::numeric_limits<unsigned int>::max());

  temp_memory->LazyAllocate(sizeof(CoordinateT) * (num_tiles + 1));
  CoordinateT *tmp_tile_coordinates =
      reinterpret_cast<CoordinateT *>(temp_memory->d_temp_storage);

  FindMergePartitions(device_idx, tmp_tile_coordinates, num_tiles,
                      BLOCK_THREADS, segments, num_segments, count);

  LbsKernel<TILE_SIZE, ITEMS_PER_THREAD, BLOCK_THREADS, OffsetT>
      <<<uint32_t(num_tiles), BLOCK_THREADS>>>(tmp_tile_coordinates,  // NOLINT
                                               segments + 1, f, num_segments);
}

template <typename FunctionT, typename OffsetT>
void DenseTransformLbs(int device_idx, OffsetT count, OffsetT num_segments,
                       FunctionT f) {
  CHECK(count % num_segments == 0) << "Data is not dense.";

  LaunchN(device_idx, count, [=] __device__(OffsetT idx) {
    OffsetT segment = idx / (count / num_segments);
    f(idx, segment);
  });
}

/**
 * \fn  template <typename FunctionT, typename SegmentIterT, typename OffsetT>
 * void TransformLbs(int device_idx, dh::CubMemory *temp_memory, OffsetT count,
 * SegmentIterT segments, OffsetT num_segments, bool is_dense, FunctionT f)
 *
 * \brief Load balancing search function. Reads a CSR type matrix description
 * and allows a function to be executed on each element. Search 'modern GPU load
 * balancing search' for more information.
 *
 * \author  Rory
 * \date  7/9/2017
 *
 * \tparam  FunctionT        Type of the function t.
 * \tparam  SegmentIterT Type of the segments iterator.
 * \tparam  OffsetT      Type of the offset.
 * \param           device_idx    Zero-based index of the device.
 * \param [in,out]  temp_memory   Temporary memory allocator.
 * \param           count         Number of elements.
 * \param           segments      Device pointer to segments.
 * \param           num_segments  Number of segments.
 * \param           is_dense      True if this object is dense.
 * \param           f             Lambda to be executed on matrix elements.
 */

template <typename FunctionT, typename SegmentIterT, typename OffsetT>
void TransformLbs(int device_idx, dh::CubMemory *temp_memory, OffsetT count,
                  SegmentIterT segments, OffsetT num_segments, bool is_dense,
                  FunctionT f) {
  if (is_dense) {
    DenseTransformLbs(device_idx, count, num_segments, f);
  } else {
    SparseTransformLbs(device_idx, temp_memory, count, segments, num_segments,
                       f);
  }
}

/**
 * @brief Helper function to sort the pairs using cub's segmented RadixSortPairs
 * @param tmp_mem cub temporary memory info
 * @param keys keys double-buffer array
 * @param vals the values double-buffer array
 * @param nVals number of elements in the array
 * @param nSegs number of segments
 * @param offsets the segments
 */
template <typename T1, typename T2>
void SegmentedSort(dh::CubMemory *tmp_mem, dh::DoubleBuffer<T1> *keys,
                   dh::DoubleBuffer<T2> *vals, int nVals, int nSegs,
                   xgboost::common::Span<int> offsets, int start = 0,
                   int end = sizeof(T1) * 8) {
  size_t tmpSize;
  dh::safe_cuda(cub::DeviceSegmentedRadixSort::SortPairs(
      NULL, tmpSize, keys->CubBuffer(), vals->CubBuffer(), nVals, nSegs,
      offsets.data(), offsets.data() + 1, start, end));
  tmp_mem->LazyAllocate(tmpSize);
  dh::safe_cuda(cub::DeviceSegmentedRadixSort::SortPairs(
      tmp_mem->d_temp_storage, tmpSize, keys->CubBuffer(), vals->CubBuffer(),
      nVals, nSegs, offsets.data(), offsets.data() + 1, start, end));
}

/**
 * @brief Helper function to perform device-wide sum-reduction
 * @param tmp_mem cub temporary memory info
 * @param in the input array to be reduced
 * @param out the output reduced value
 * @param nVals number of elements in the input array
 */
template <typename T>
void SumReduction(dh::CubMemory* tmp_mem, xgboost::common::Span<T> in, xgboost::common::Span<T> out,
                  int nVals) {
  size_t tmpSize;
  dh::safe_cuda(
      cub::DeviceReduce::Sum(NULL, tmpSize, in.data(), out.data(), nVals));
  tmp_mem->LazyAllocate(tmpSize);
  dh::safe_cuda(cub::DeviceReduce::Sum(tmp_mem->d_temp_storage, tmpSize,
                                       in.data(), out.data(), nVals));
}

/**
* @brief Helper function to perform device-wide sum-reduction, returns to the
* host
* @param tmp_mem cub temporary memory info
* @param in the input array to be reduced
* @param nVals number of elements in the input array
*/
template <typename T>
typename std::iterator_traits<T>::value_type SumReduction(
    dh::CubMemory* tmp_mem, T in, int nVals) {
  using ValueT = typename std::iterator_traits<T>::value_type;
  size_t tmpSize {0};
  ValueT *dummy_out = nullptr;
  dh::safe_cuda(cub::DeviceReduce::Sum(nullptr, tmpSize, in, dummy_out, nVals));
  // Allocate small extra memory for the return value
  tmp_mem->LazyAllocate(tmpSize + sizeof(ValueT));
  auto ptr = reinterpret_cast<ValueT *>(tmp_mem->d_temp_storage) + 1;
  dh::safe_cuda(cub::DeviceReduce::Sum(
      reinterpret_cast<void *>(ptr), tmpSize, in,
      reinterpret_cast<ValueT *>(tmp_mem->d_temp_storage),
      nVals));
  ValueT sum;
  dh::safe_cuda(cudaMemcpy(&sum, tmp_mem->d_temp_storage, sizeof(ValueT),
                           cudaMemcpyDeviceToHost));
  return sum;
}

/**
 * @brief Fill a given constant value across all elements in the buffer
 * @param out the buffer to be filled
 * @param len number of elements i the buffer
 * @param def default value to be filled
 */
template <typename T, int BlkDim = 256, int ItemsPerThread = 4>
void FillConst(int device_idx, T *out, int len, T def) {
  dh::LaunchN<ItemsPerThread, BlkDim>(device_idx, len,
                                      [=] __device__(int i) { out[i] = def; });
}

/**
 * @brief gather elements
 * @param out1 output gathered array for the first buffer
 * @param in1 first input buffer
 * @param out2 output gathered array for the second buffer
 * @param in2 second input buffer
 * @param instId gather indices
 * @param nVals length of the buffers
 */
template <typename T1, typename T2, int BlkDim = 256, int ItemsPerThread = 4>
void Gather(int device_idx, T1 *out1, const T1 *in1, T2 *out2, const T2 *in2,
            const int *instId, int nVals) {
  dh::LaunchN<ItemsPerThread, BlkDim>(device_idx, nVals,
                                       [=] __device__(int i) {
                                         int iid = instId[i];
                                         T1 v1 = in1[iid];
                                         T2 v2 = in2[iid];
                                         out1[i] = v1;
                                         out2[i] = v2;
                                       });
}

/**
 * @brief gather elements
 * @param out output gathered array
 * @param in input buffer
 * @param instId gather indices
 * @param nVals length of the buffers
 */
template <typename T, int BlkDim = 256, int ItemsPerThread = 4>
void Gather(int device_idx, T *out, const T *in, const int *instId, int nVals) {
  dh::LaunchN<ItemsPerThread, BlkDim>(device_idx, nVals,
                                       [=] __device__(int i) {
                                         int iid = instId[i];
                                         out[i] = in[iid];
                                       });
}

class SaveCudaContext {
 private:
  int saved_device_;

 public:
  template <typename Functor>
  explicit SaveCudaContext (Functor func) : saved_device_{-1} {
    // When compiled with CUDA but running on CPU only device,
    // cudaGetDevice will fail.
    try {
      safe_cuda(cudaGetDevice(&saved_device_));
    } catch (const dmlc::Error &except) {
      saved_device_ = -1;
    }
    func();
  }
  ~SaveCudaContext() {
    if (saved_device_ != -1) {
      safe_cuda(cudaSetDevice(saved_device_));
    }
  }
};

/**
 * \class AllReducer
 *
 * \brief All reducer class that manages its own communication group and
 * streams. Must be initialised before use. If XGBoost is compiled without NCCL
 * this is a dummy class that will error if used with more than one GPU.
 */
class AllReducer {
  bool initialised_ {false};
  size_t allreduce_bytes_ {0};  // Keep statistics of the number of bytes communicated
  size_t allreduce_calls_ {0};  // Keep statistics of the number of reduce calls
  std::vector<size_t> host_data_;  // Used for all reduce on host
#ifdef XGBOOST_USE_NCCL
  ncclComm_t comm_;
  cudaStream_t stream_;
  int device_ordinal_;
  ncclUniqueId id_;
#endif

 public:
  AllReducer() = default;

  /**
   * \brief Initialise with the desired device ordinal for this communication
   * group.
   *
   * \param device_ordinal The device ordinal.
   */
  void Init(int _device_ordinal);

  ~AllReducer();

  /**
   * \brief Allreduce. Use in exactly the same way as NCCL but without needing
   * streams or comms.
   *
   * \param sendbuff                The sendbuff.
   * \param recvbuff                The recvbuff.
   * \param count                   Number of elements.
   */

  void AllReduceSum(const double *sendbuff, double *recvbuff, int count) {
#ifdef XGBOOST_USE_NCCL
    CHECK(initialised_);
    dh::safe_cuda(cudaSetDevice(device_ordinal_));
    dh::safe_nccl(ncclAllReduce(sendbuff, recvbuff, count, ncclDouble, ncclSum, comm_, stream_));
    allreduce_bytes_ += count * sizeof(double);
    allreduce_calls_ += 1;
#endif
  }

  /**
   * \brief Allreduce. Use in exactly the same way as NCCL but without needing
   * streams or comms.
   *
   * \param sendbuff                The sendbuff.
   * \param recvbuff                The recvbuff.
   * \param count                   Number of elements.
   */

  void AllReduceSum(const float *sendbuff, float *recvbuff, int count) {
#ifdef XGBOOST_USE_NCCL
    CHECK(initialised_);
    dh::safe_cuda(cudaSetDevice(device_ordinal_));
    dh::safe_nccl(ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm_, stream_));
    allreduce_bytes_ += count * sizeof(float);
    allreduce_calls_ += 1;
#endif
  }

  /**
   * \brief Allreduce. Use in exactly the same way as NCCL but without needing streams or comms.
   *
   * \param count Number of.
   *
   * \param sendbuff                The sendbuff.
   * \param recvbuff                The recvbuff.
   * \param count                   Number of.
   */

  void AllReduceSum(const int64_t *sendbuff, int64_t *recvbuff, int count) {
#ifdef XGBOOST_USE_NCCL
    CHECK(initialised_);

    dh::safe_cuda(cudaSetDevice(device_ordinal_));
    dh::safe_nccl(ncclAllReduce(sendbuff, recvbuff, count, ncclInt64, ncclSum, comm_, stream_));
#endif
  }

  /**
   * \fn  void Synchronize()
   *
   * \brief Synchronizes the entire communication group.
   */
  void Synchronize() {
#ifdef XGBOOST_USE_NCCL
    dh::safe_cuda(cudaSetDevice(device_ordinal_));
    dh::safe_cuda(cudaStreamSynchronize(stream_));
#endif
  };

#ifdef XGBOOST_USE_NCCL
  /**
   * \fn  ncclUniqueId GetUniqueId()
   *
   * \brief Gets the Unique ID from NCCL to be used in setting up interprocess
   * communication
   *
   * \return the Unique ID
   */
  ncclUniqueId GetUniqueId() {
    static const int kRootRank = 0;
    ncclUniqueId id;
    if (rabit::GetRank() == kRootRank) {
      dh::safe_nccl(ncclGetUniqueId(&id));
    }
    rabit::Broadcast(
        static_cast<void*>(&id),
        sizeof(ncclUniqueId),
        static_cast<int>(kRootRank));
    return id;
  }
#endif
  /** \brief Perform max all reduce operation on the host. This function first
   * reduces over omp threads then over nodes using rabit (which is not thread
   * safe) using the master thread. Uses naive reduce algorithm for local
   * threads, don't expect this to scale.*/
  void HostMaxAllReduce(std::vector<size_t> *p_data) {
#ifdef XGBOOST_USE_NCCL
    auto &data = *p_data;
    // Wait in case some other thread is accessing host_data_
#pragma omp barrier
    // Reset shared buffer
#pragma omp single
    {
      host_data_.resize(data.size());
      std::fill(host_data_.begin(), host_data_.end(), size_t(0));
    }
    // Threads update shared array
    for (auto i = 0ull; i < data.size(); i++) {
#pragma omp critical
      { host_data_[i] = std::max(host_data_[i], data[i]); }
    }
    // Wait until all threads are finished
#pragma omp barrier

    // One thread performs all reduce across distributed nodes
#pragma omp master
    {
      rabit::Allreduce<rabit::op::Max, size_t>(host_data_.data(),
                                               host_data_.size());
    }

#pragma omp barrier

    // Threads can now read back all reduced values
    for (auto i = 0ull; i < data.size(); i++) {
      data[i] = host_data_[i];
    }
#endif
  }
};

template <typename T,
  typename IndexT = typename xgboost::common::Span<T>::index_type>
xgboost::common::Span<T> ToSpan(
    device_vector<T>& vec,
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
thrust::device_ptr<T> tbegin(xgboost::common::Span<T>& span) {  // NOLINT
  return thrust::device_ptr<T>(span.data());
}

template <typename T>
thrust::device_ptr<T> tend(xgboost::common::Span<T>& span) {  // NOLINT
  return tbegin(span) + span.size();
}

template <typename T>
thrust::device_ptr<T const> tcbegin(xgboost::common::Span<T> const& span) {  // NOLINT
  return thrust::device_ptr<T const>(span.data());
}

template <typename T>
thrust::device_ptr<T const> tcend(xgboost::common::Span<T> const& span) {  // NOLINT
  return tcbegin(span) + span.size();
}

// This type sorts an array which is divided into multiple groups. The sorting is influenced
// by the function object 'Comparator'
template <typename T>
class SegmentSorter {
 private:
  // Items sorted within the group
  caching_device_vector<T> ditems_;

  // Original position of the items before they are sorted descendingly within its groups
  caching_device_vector<uint32_t> doriginal_pos_;

  // Segments within the original list that delineates the different groups
  caching_device_vector<uint32_t> group_segments_;

  // Need this on the device as it is used in the kernels
  caching_device_vector<uint32_t> dgroups_;       // Group information on device

  // Where did the item that was originally present at position 'x' move to after they are sorted
  caching_device_vector<uint32_t> dindexable_sorted_pos_;

  // Initialize everything but the segments
  void Init(uint32_t num_elems) {
    ditems_.resize(num_elems);

    doriginal_pos_.resize(num_elems);
    thrust::sequence(doriginal_pos_.begin(), doriginal_pos_.end());
  }

  // Initialize all with group info
  void Init(const std::vector<uint32_t> &groups) {
    uint32_t num_elems = groups.back();
    this->Init(num_elems);
    this->CreateGroupSegments(groups);
  }

 public:
  // This needs to be public due to device lambda
  void CreateGroupSegments(const std::vector<uint32_t> &groups) {
    uint32_t num_elems = groups.back();
    group_segments_.resize(num_elems, 0);

    dgroups_ = groups;

    if (GetNumGroups() == 1) return;  // There are no segments; hence, no need to compute them

    // Define the segments by assigning a group ID to each element
    const uint32_t *dgroups = dgroups_.data().get();
    uint32_t ngroups = dgroups_.size();
    auto ComputeGroupIDLambda = [=] __device__(uint32_t idx) {
      return dh::UpperBound(dgroups, ngroups, idx) - 1;
    };  // NOLINT

    thrust::transform(thrust::make_counting_iterator(static_cast<uint32_t>(0)),
                      thrust::make_counting_iterator(num_elems),
                      group_segments_.begin(),
                      ComputeGroupIDLambda);
  }

  // Accessors that returns device pointer
  inline uint32_t GetNumItems() const { return ditems_.size(); }
  inline const xgboost::common::Span<const T> GetItemsSpan() const {
    return { ditems_.data().get(), ditems_.size() };
  }

  inline const xgboost::common::Span<const uint32_t> GetOriginalPositionsSpan() const {
    return { doriginal_pos_.data().get(), doriginal_pos_.size() };
  }

  inline const xgboost::common::Span<const uint32_t> GetGroupSegmentsSpan() const {
    return { group_segments_.data().get(), group_segments_.size() };
  }

  inline uint32_t GetNumGroups() const { return dgroups_.size() - 1; }
  inline const xgboost::common::Span<const uint32_t> GetGroupsSpan() const {
    return { dgroups_.data().get(), dgroups_.size() };
  }

  inline const xgboost::common::Span<const uint32_t> GetIndexableSortedPositionsSpan() const {
    return { dindexable_sorted_pos_.data().get(), dindexable_sorted_pos_.size() };
  }

  // Sort an array that is divided into multiple groups. The array is sorted within each group.
  // This version provides the group information that is on the host.
  // The array is sorted based on an adaptable binary predicate. By default a stateless predicate
  // is used.
  template <typename Comparator = thrust::greater<T>>
  void SortItems(const T *ditems, uint32_t item_size, const std::vector<uint32_t> &groups,
                 const Comparator &comp = Comparator()) {
    this->Init(groups);
    this->SortItems(ditems, item_size, this->GetGroupSegmentsSpan(), comp);
  }

  // Sort an array that is divided into multiple groups. The array is sorted within each group.
  // This version provides the group information that is on the device.
  // The array is sorted based on an adaptable binary predicate. By default a stateless predicate
  // is used.
  template <typename Comparator = thrust::greater<T>>
  void SortItems(const T *ditems, uint32_t item_size,
                 const xgboost::common::Span<const uint32_t> &group_segments,
                 const Comparator &comp = Comparator()) {
    this->Init(item_size);

    // Sort the items that are grouped. We would like to avoid using predicates to perform the sort,
    // as thrust resorts to using a merge sort as opposed to a much much faster radix sort
    // when comparators are used. Hence, the following algorithm is used. This is done so that
    // we can grab the appropriate related values from the original list later, after the
    // items are sorted.
    //
    // Here is the internal representation:
    // dgroups_:          [ 0, 3, 5, 8, 10 ]
    // group_segments_:   0 0 0 | 1 1 | 2 2 2 | 3 3
    // doriginal_pos_:    0 1 2 | 3 4 | 5 6 7 | 8 9
    // ditems_:           1 0 1 | 2 1 | 1 3 3 | 4 4 (from original items)
    //
    // Sort the items first and make a note of the original positions in doriginal_pos_
    // based on the sort
    // ditems_:           4 4 3 3 2 1 1 1 1 0
    // doriginal_pos_:    8 9 6 7 3 0 2 4 5 1
    // NOTE: This consumes space, but is much faster than some of the other approaches - sorting
    //       in kernel, sorting using predicates etc.

    ditems_.assign(thrust::device_ptr<const T>(ditems),
                   thrust::device_ptr<const T>(ditems) + item_size);

    // Allocator to be used by sort for managing space overhead while sorting
    dh::XGBCachingDeviceAllocator<char> alloc;

    thrust::stable_sort_by_key(thrust::cuda::par(alloc),
                               ditems_.begin(), ditems_.end(),
                               doriginal_pos_.begin(), comp);

    if (GetNumGroups() == 1) return;  // The entire array is sorted, as it isn't segmented

    // Next, gather the segments based on the doriginal_pos_. This is to reflect the
    // holisitic item sort order on the segments
    // group_segments_c_:   3 3 2 2 1 0 0 1 2 0
    // doriginal_pos_:      8 9 6 7 3 0 2 4 5 1 (stays the same)
    caching_device_vector<uint32_t> group_segments_c(item_size);
    thrust::gather(doriginal_pos_.begin(), doriginal_pos_.end(),
                   dh::tcbegin(group_segments), group_segments_c.begin());

    // Now, sort the group segments so that you may bring the items within the group together,
    // in the process also noting the relative changes to the doriginal_pos_ while that happens
    // group_segments_c_:   0 0 0 1 1 2 2 2 3 3
    // doriginal_pos_:      0 2 1 3 4 6 7 5 8 9
    thrust::stable_sort_by_key(thrust::cuda::par(alloc),
                               group_segments_c.begin(), group_segments_c.end(),
                               doriginal_pos_.begin(), thrust::less<uint32_t>());

    // Finally, gather the original items based on doriginal_pos_ to sort the input and
    // to store them in ditems_
    // doriginal_pos_:      0 2 1 3 4 6 7 5 8 9  (stays the same)
    // ditems_:             1 1 0 2 1 3 3 1 4 4  (from unsorted items - ditems)
    thrust::gather(doriginal_pos_.begin(), doriginal_pos_.end(),
                   thrust::device_ptr<const T>(ditems), ditems_.begin());
  }

  // Determine where an item that was originally present at position 'x' has been relocated to
  // after a sort. Creation of such an index has to be explicitly requested after a sort
  void CreateIndexableSortedPositions() {
    dindexable_sorted_pos_.resize(GetNumItems());
    thrust::scatter(thrust::make_counting_iterator(static_cast<uint32_t>(0)),
                    thrust::make_counting_iterator(GetNumItems()),  // Rearrange indices...
                    // ...based on this map
                    dh::tcbegin(GetOriginalPositionsSpan()),
                    dindexable_sorted_pos_.begin());  // Write results into this
  }
};

template <typename FunctionT>
class LauncherItr {
public:
  int idx { 0 };
  FunctionT f;
  XGBOOST_DEVICE LauncherItr() : idx(0) {}  // NOLINT
  XGBOOST_DEVICE LauncherItr(int idx, FunctionT f) : idx(idx), f(f) {}
  XGBOOST_DEVICE LauncherItr &operator=(int output) {
    f(idx, output);
    return *this;
  }
};

/**
 * \brief Thrust compatible iterator type - discards algorithm output and launches device lambda
 *        with the index of the output and the algorithm output as arguments.
 *
 * \author  Rory
 * \date  7/9/2017
 *
 * \tparam  FunctionT Type of the function t.
 */
template <typename FunctionT>
class DiscardLambdaItr {
public:
 // Required iterator traits
 using self_type = DiscardLambdaItr;  // NOLINT
 using difference_type = ptrdiff_t;   // NOLINT
 using value_type = void;       // NOLINT
 using pointer = value_type *;  // NOLINT
 using reference = LauncherItr<FunctionT>;  // NOLINT
 using iterator_category = typename thrust::detail::iterator_facade_category<  // NOLINT
     thrust::any_system_tag, thrust::random_access_traversal_tag, value_type,
     reference>::type;  // NOLINT
private:
  difference_type offset_;
  FunctionT f_;
public:
 XGBOOST_DEVICE explicit DiscardLambdaItr(FunctionT f) : offset_(0), f_(f) {}
 XGBOOST_DEVICE DiscardLambdaItr(difference_type offset, FunctionT f)
     : offset_(offset), f_(f) {}
 XGBOOST_DEVICE self_type operator+(const int &b) const {
   return DiscardLambdaItr(offset_ + b, f_);
  }
  XGBOOST_DEVICE self_type operator++() {
    offset_++;
    return *this;
  }
  XGBOOST_DEVICE self_type operator++(int) {
    self_type retval = *this;
    offset_++;
    return retval;
  }
  XGBOOST_DEVICE self_type &operator+=(const int &b) {
    offset_ += b;
    return *this;
  }
  XGBOOST_DEVICE reference operator*() const {
    return LauncherItr<FunctionT>(offset_, f_);
  }
  XGBOOST_DEVICE reference operator[](int idx) {
    self_type offset = (*this) + idx;
    return *offset;
  }
};

// Atomic add function for gradients
template <typename OutputGradientT, typename InputGradientT>
DEV_INLINE void AtomicAddGpair(OutputGradientT* dest,
                               const InputGradientT& gpair) {
  auto dst_ptr = reinterpret_cast<typename OutputGradientT::ValueT*>(dest);

  atomicAdd(dst_ptr,
            static_cast<typename OutputGradientT::ValueT>(gpair.GetGrad()));
  atomicAdd(dst_ptr + 1,
            static_cast<typename OutputGradientT::ValueT>(gpair.GetHess()));
}


// Thrust version of this function causes error on Windows
template <typename ReturnT, typename IterT, typename FuncT>
thrust::transform_iterator<FuncT, IterT, ReturnT> MakeTransformIterator(
  IterT iter, FuncT func) {
  return thrust::transform_iterator<FuncT, IterT, ReturnT>(iter, func);
}

}  // namespace dh
