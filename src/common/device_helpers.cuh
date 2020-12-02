/*!
 * Copyright 2017-2020 XGBoost contributors
 */
#pragma once
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
#include <thrust/execution_policy.h>

#include <thrust/transform_scan.h>
#include <thrust/logical.h>
#include <thrust/gather.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>

#include <rabit/rabit.h>
#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <tuple>

#include "xgboost/logging.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/span.h"

#include "common.h"

#ifdef XGBOOST_USE_NCCL
#include "nccl.h"
#endif  // XGBOOST_USE_NCCL

#if defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1
#include "rmm/mr/device/per_device_resource.hpp"
#include "rmm/mr/device/thrust_allocator_adaptor.hpp"
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
T __device__ __forceinline__ atomicAdd(T *addr, T v) {  // NOLINT
  using Type = typename dh::detail::AtomicDispatcher<sizeof(T)>::Type;
  Type ret = ::atomicAdd(reinterpret_cast<Type *>(addr), static_cast<Type>(v));
  return static_cast<T>(ret);
}

namespace dh {

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
  return size_t(max_shared_memory);
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
  return size_t(max_shared_memory);
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
        LOG(WARNING) << "Attempting to deallocate " << n << " bytes on device "
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
    auto rank = rabit::GetRank();
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
  auto rank = rabit::GetRank();
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
    : SuperT(rmm::mr::get_current_device_resource(), cudaStream_t{nullptr}) {}
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
    static cub::CachingDeviceAllocator *allocator = new cub::CachingDeviceAllocator(2, 9, 29);
    return *allocator;
  }
  pointer allocate(size_t n) {  // NOLINT
    T* ptr;
    auto errc =  GetGlobalCachingAllocator().DeviceAllocate(reinterpret_cast<void **>(&ptr),
                                                            n * sizeof(T));
    if (errc != cudaSuccess) {
      ThrowOOMError("Caching allocator", n * sizeof(T));
    }
    pointer thrust_ptr{ ptr };
    GlobalMemoryLogger().RegisterAllocation(thrust_ptr.get(), n * sizeof(T));
    return thrust_ptr;
  }
  void deallocate(pointer ptr, size_t n) {  // NOLINT
    GlobalMemoryLogger().RegisterDeallocation(ptr.get(), n * sizeof(T));
    GetGlobalCachingAllocator().DeviceFree(ptr.get());
  }
  XGBOOST_DEVICE void construct(T *) {}  // NOLINT
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
    LaunchN(device, this->size(), [=] __device__(size_t idx) { d_data[idx] = val; });
  }
  thrust::device_ptr<T> data() { return ptr_; }  // NOLINT
  size_t size() { return size_; }  // NOLINT

 private:
  thrust::device_ptr<T> ptr_;
  size_t size_;
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
   * \brief Allgather implemented as grouped calls to Broadcast.  This way we can accept
   *        different size of data on different workers.
   * \param length_bytes Size of input data in bytes.
   * \param segments     Size of data on each worker.
   * \param recvbuf      Buffer storing the result of data from all workers.
   */
  void AllGather(void const* data, size_t length_bytes,
                 std::vector<size_t>* segments, dh::caching_device_vector<char>* recvbuf);

  void AllGather(uint32_t const* data, size_t length,
                 dh::caching_device_vector<uint32_t>* recvbuf) {
#ifdef XGBOOST_USE_NCCL
    CHECK(initialised_);
    size_t world = rabit::GetWorldSize();
    recvbuf->resize(length * world);
    safe_nccl(ncclAllGather(data, recvbuf->data().get(), length, ncclUint32,
                            comm_, stream_));
#endif  // XGBOOST_USE_NCCL
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

  void AllReduceSum(const uint32_t *sendbuff, uint32_t *recvbuff, int count) {
#ifdef XGBOOST_USE_NCCL
    CHECK(initialised_);

    dh::safe_cuda(cudaSetDevice(device_ordinal_));
    dh::safe_nccl(ncclAllReduce(sendbuff, recvbuff, count, ncclUint32, ncclSum, comm_, stream_));
#endif
  }

  void AllReduceSum(const uint64_t *sendbuff, uint64_t *recvbuff, int count) {
#ifdef XGBOOST_USE_NCCL
    CHECK(initialised_);

    dh::safe_cuda(cudaSetDevice(device_ordinal_));
    dh::safe_nccl(ncclAllReduce(sendbuff, recvbuff, count, ncclUint64, ncclSum, comm_, stream_));
#endif
  }

  // Specialization for size_t, which is implementation defined so it might or might not
  // be one of uint64_t/uint32_t/unsigned long long/unsigned long.
  template <typename T = size_t,
            std::enable_if_t<std::is_same<size_t, T>::value &&
                             !std::is_same<size_t, unsigned long long>::value>  // NOLINT
                * = nullptr>
  void AllReduceSum(const T *sendbuff, T *recvbuff, int count) { // NOLINT
#ifdef XGBOOST_USE_NCCL
    CHECK(initialised_);

    dh::safe_cuda(cudaSetDevice(device_ordinal_));
    static_assert(sizeof(unsigned long long) == sizeof(uint64_t), ""); // NOLINT
    dh::safe_nccl(ncclAllReduce(sendbuff, recvbuff, count, ncclUint64, ncclSum, comm_, stream_));
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
};

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
      return thrust::upper_bound(thrust::seq, dgroups, dgroups + ngroups, idx) -
             dgroups - 1;
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
  size_t segment_id = thrust::upper_bound(thrust::seq, first, last, idx) -
                      1 - first;
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
          typename ValOutIt, typename Comp>
size_t
SegmentedUnique(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                KeyInIt key_segments_first, KeyInIt key_segments_last, ValInIt val_first,
                ValInIt val_last, KeyOutIt key_segments_out, ValOutIt val_out,
                Comp comp) {
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
        if (l.first == r.first) {
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
  return SegmentedUnique(thrust::cuda::par(alloc), std::forward<Inputs&&>(inputs)...);
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
}  // namespace dh
