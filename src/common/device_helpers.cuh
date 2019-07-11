/*!
 * Copyright 2017-2019 XGBoost contributors
 */
#pragma once
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
#include <xgboost/logging.h>
#include <rabit/rabit.h>
#include <cub/util_allocator.cuh>

#include "common.h"
#include "span.h"

#include <algorithm>
#include <omp.h>
#include <chrono>
#include <ctime>
#include <cub/cub.cuh>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include "timer.h"

#ifdef XGBOOST_USE_NCCL
#include "nccl.h"
#include "../common/io.h"
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

inline void CudaCheckPointerDevice(void* ptr) {
  cudaPointerAttributes attr;
  dh::safe_cuda(cudaPointerGetAttributes(&attr, ptr));
  int ptr_device = attr.device;
  int cur_device = -1;
  cudaGetDevice(&cur_device);
  CHECK_EQ(ptr_device, cur_device) << "pointer device: " << ptr_device
                                   << "current device: " << cur_device;
}

template <typename T>
const T *Raw(const thrust::device_vector<T> &v) {  //  NOLINT
  return raw_pointer_cast(v.data());
}

// if n_devices=-1, then use all visible devices
inline void SynchronizeNDevices(xgboost::GPUSet devices) {
  devices = devices.IsEmpty() ? xgboost::GPUSet::AllVisible() : devices;
  for (auto const d : devices) {
    safe_cuda(cudaSetDevice(d));
    safe_cuda(cudaDeviceSynchronize());
  }
}

inline void SynchronizeAll() {
  for (int device_idx : xgboost::GPUSet::AllVisible()) {
    safe_cuda(cudaSetDevice(device_idx));
    safe_cuda(cudaDeviceSynchronize());
  }
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
  for (int d_idx : xgboost::GPUSet::AllVisible()) {
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
  atomicOr(&buffer[ibyte / sizeof(unsigned int)], (unsigned int)b << (ibyte % (sizeof(unsigned int)) * 8));
}

/*!
 * \brief Find the strict upper bound for an element in a sorted array
 *  using binary search.
 * \param cuts pointer to the first element of the sorted array
 * \param n length of the sorted array
 * \param v value for which to find the upper bound
 * \return the smallest index i such that v < cuts[i], or n if v is greater or equal
 *  than all elements of the array
*/
DEV_INLINE int UpperBound(const float* __restrict__ cuts, int n, float v) {
  if (n == 0)           { return 0; }
  if (cuts[n - 1] <= v) { return n; }
  if (cuts[0] > v)      { return 0; }

  int left = 0, right = n - 1;
  while (right - left > 1) {
    int middle = left + (right - left) / 2;
    if (cuts[middle] > v) {
      right = middle;
    } else {
      left = middle;
    }
  }
  return right;
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

template <int ITEMS_PER_THREAD = 8, int BLOCK_THREADS = 256, typename L>
inline void LaunchN(int device_idx, size_t n, cudaStream_t stream, L lambda) {
  if (n == 0) {
    return;
  }

  safe_cuda(cudaSetDevice(device_idx));

  const int GRID_SIZE =
      static_cast<int>(xgboost::common::DivRoundUp(n, ITEMS_PER_THREAD * BLOCK_THREADS));
  LaunchNKernel<<<GRID_SIZE, BLOCK_THREADS, 0, stream>>>(static_cast<size_t>(0),
                                                         n, lambda);
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
      CHECK_EQ(itr->second, n);
      currently_allocated_bytes -= itr->second;
      device_allocations.erase(itr);
    }
  };
  std::map<int, DeviceStats>
    stats_;  // Map device ordinal to memory information
  std::mutex mutex_;

public:
  void RegisterAllocation(void *ptr, size_t n) {
    if (!xgboost::ConsoleLogger::ShouldLog(xgboost::ConsoleLogger::LV::kDebug))
      return;
    std::lock_guard<std::mutex> guard(mutex_);
    int current_device;
    safe_cuda(cudaGetDevice(&current_device));
    stats_[current_device].RegisterAllocation(ptr, n);
    CHECK_LE(stats_[current_device].peak_allocated_bytes, dh::TotalMemory(current_device));
  }
  void RegisterDeallocation(void *ptr, size_t n) {
    if (!xgboost::ConsoleLogger::ShouldLog(xgboost::ConsoleLogger::LV::kDebug))
      return;
    std::lock_guard<std::mutex> guard(mutex_);
    int current_device;
    safe_cuda(cudaGetDevice(&current_device));
    stats_[current_device].RegisterDeallocation(ptr, n, current_device);
  }
  void Log() {
    if (!xgboost::ConsoleLogger::ShouldLog(xgboost::ConsoleLogger::LV::kDebug))
      return;
    std::lock_guard<std::mutex> guard(mutex_);
    for (const auto &kv : stats_) {
      LOG(CONSOLE) << "======== Device " << kv.first << " Memory Allocations: "
        << " ========";
      LOG(CONSOLE) << "Peak memory usage: "
        << kv.second.peak_allocated_bytes / 1000000 << "mb";
      LOG(CONSOLE) << "Number of allocations: " << kv.second.num_allocations;
    }
  }
};
};

inline detail::MemoryLogger &GlobalMemoryLogger() {
  static detail::MemoryLogger memory_logger;
  return memory_logger;
}

namespace detail{
/**
 * \brief Default memory allocator, uses cudaMalloc/Free and logs allocations if verbose.
 */
template <class T>
struct XGBDefaultDeviceAllocatorImpl : thrust::device_malloc_allocator<T> {
  using super_t = thrust::device_malloc_allocator<T>;
  using pointer = thrust::device_ptr<T>;
  template<typename U>
  struct rebind
  {
    typedef XGBDefaultDeviceAllocatorImpl<U> other;
  };
  pointer allocate(size_t n) {
    pointer ptr = super_t::allocate(n);
    GlobalMemoryLogger().RegisterAllocation(ptr.get(), n * sizeof(T));
    return ptr;
  }
  void deallocate(pointer ptr, size_t n) {
    GlobalMemoryLogger().RegisterDeallocation(ptr.get(), n * sizeof(T));
    return super_t::deallocate(ptr, n);
  }
};

/**
 * \brief Caching memory allocator, uses cub::CachingDeviceAllocator as a back-end and logs allocations if verbose. Does not initialise memory on construction.
 */
template <class T>
struct XGBCachingDeviceAllocatorImpl : thrust::device_malloc_allocator<T> {
  using pointer = thrust::device_ptr<T>;
  template<typename U>
  struct rebind
  {
    typedef XGBCachingDeviceAllocatorImpl<U> other;
  };
   cub::CachingDeviceAllocator& GetGlobalCachingAllocator ()
   {
    // Configure allocator with maximum cached bin size of ~1GB and no limit on
    // maximum cached bytes
     static cub::CachingDeviceAllocator *allocator = new cub::CachingDeviceAllocator(2, 9, 29);
     return *allocator;
   }
   pointer allocate(size_t n) {
     T *ptr;
     GetGlobalCachingAllocator().DeviceAllocate(reinterpret_cast<void **>(&ptr),
                                                n * sizeof(T));
     pointer thrust_ptr(ptr);
     GlobalMemoryLogger().RegisterAllocation(thrust_ptr.get(), n * sizeof(T));
     return thrust_ptr;
   }
   void deallocate(pointer ptr, size_t n) {
     GlobalMemoryLogger().RegisterDeallocation(ptr.get(), n * sizeof(T));
     GetGlobalCachingAllocator().DeviceFree(ptr.get());
   }
  __host__ __device__
    void construct(T *)
  {
    // no-op
  }
};
};

// Declare xgboost allocators
// Replacement of allocator with custom backend should occur here
template <typename T>
using XGBDeviceAllocator = detail::XGBDefaultDeviceAllocatorImpl<T>;
template <typename T>
using XGBCachingDeviceAllocator = detail::XGBCachingDeviceAllocatorImpl<T>;
/** \brief Specialisation of thrust device vector using custom allocator. */
template <typename T>
using device_vector = thrust::device_vector<T,  XGBDeviceAllocator<T>>;
template <typename T>
using caching_device_vector = thrust::device_vector<T,  XGBCachingDeviceAllocator<T>>;

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
    return xgboost::common::Span<T>{
        buff.Current(),
        static_cast<typename xgboost::common::Span<T>::index_type>(Size())};
  }

  T *other() { return buff.Alternate(); }
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
  std::vector<int> device_idx_;

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

  ~BulkAllocator() {
    for (size_t i = 0; i < d_ptr_.size(); i++) {
      if (!(d_ptr_[i] == nullptr)) {
        safe_cuda(cudaSetDevice(device_idx_[i]));
        XGBDeviceAllocator<char> allocator;
        allocator.deallocate(thrust::device_ptr<char>(d_ptr_[i]), size_[i]);
        d_ptr_[i] = nullptr;
      }
    }
  }

  // returns sum of bytes for all allocations
  size_t Size() {
    return std::accumulate(size_.begin(), size_.end(), static_cast<size_t>(0));
  }

  template <typename... Args>
  void Allocate(int device_idx, Args... args) {
    size_t size = GetSizeBytes(args...);

    char *ptr = AllocateDevice(device_idx, size);

    AllocateSpan(device_idx, ptr, args...);

    d_ptr_.push_back(ptr);
    size_.push_back(size);
    device_idx_.push_back(device_idx);
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
  void *d_temp_storage;
  size_t temp_storage_bytes;

  // Thrust
  using value_type = char;  // NOLINT

  CubMemory() : d_temp_storage(nullptr), temp_storage_bytes(0) {}

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

  typedef typename std::iterator_traits<SegmentIterT>::value_type SegmentT;
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
      <<<uint32_t(num_tiles), BLOCK_THREADS>>>(tmp_tile_coordinates,
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
void SumReduction(dh::CubMemory &tmp_mem, xgboost::common::Span<T> in, xgboost::common::Span<T> out,
                  int nVals) {
  size_t tmpSize;
  dh::safe_cuda(
      cub::DeviceReduce::Sum(NULL, tmpSize, in.data(), out.data(), nVals));
  tmp_mem.LazyAllocate(tmpSize);
  dh::safe_cuda(cub::DeviceReduce::Sum(tmp_mem.d_temp_storage, tmpSize,
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
    dh::CubMemory &tmp_mem, T in, int nVals) {
  using ValueT = typename std::iterator_traits<T>::value_type;
  size_t tmpSize {0};
  ValueT *dummy_out = nullptr;
  dh::safe_cuda(cub::DeviceReduce::Sum(nullptr, tmpSize, in, dummy_out, nVals));
  // Allocate small extra memory for the return value
  tmp_mem.LazyAllocate(tmpSize + sizeof(ValueT));
  auto ptr = reinterpret_cast<ValueT *>(tmp_mem.d_temp_storage) + 1;
  dh::safe_cuda(cub::DeviceReduce::Sum(
      reinterpret_cast<void *>(ptr), tmpSize, in,
      reinterpret_cast<ValueT *>(tmp_mem.d_temp_storage),
      nVals));
  ValueT sum;
  dh::safe_cuda(cudaMemcpy(&sum, tmp_mem.d_temp_storage, sizeof(ValueT),
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
  bool initialised_;
  size_t allreduce_bytes_;  // Keep statistics of the number of bytes communicated
  size_t allreduce_calls_;  // Keep statistics of the number of reduce calls
  std::vector<size_t> host_data;  // Used for all reduce on host
#ifdef XGBOOST_USE_NCCL
  std::vector<ncclComm_t> comms;
  std::vector<cudaStream_t> streams;
  std::vector<int> device_ordinals;  // device id from CUDA
  std::vector<int> device_counts;  // device count from CUDA
  ncclUniqueId id;
#endif

 public:
  AllReducer() : initialised_(false), allreduce_bytes_(0),
                 allreduce_calls_(0) {}

  /**
   * \brief If we are using a single GPU only
   */
  bool IsSingleGPU() {
#ifdef XGBOOST_USE_NCCL
    CHECK(device_counts.size() > 0) << "AllReducer not initialised.";
    return device_counts.size() <= 1 && device_counts.at(0) == 1;
#else
    return true;
#endif
  }

  /**
   * \brief Initialise with the desired device ordinals for this communication
   * group.
   *
   * \param device_ordinals The device ordinals.
   */

  void Init(const std::vector<int> &device_ordinals) {
#ifdef XGBOOST_USE_NCCL
    /** \brief this >monitor . init. */
    this->device_ordinals = device_ordinals;
    this->device_counts.resize(rabit::GetWorldSize());
    this->comms.resize(device_ordinals.size());
    this->streams.resize(device_ordinals.size());
    this->id = GetUniqueId();

    device_counts.at(rabit::GetRank()) = device_ordinals.size();
    for (size_t i = 0; i < device_counts.size(); i++) {
      int dev_count = device_counts.at(i);
      rabit::Allreduce<rabit::op::Sum, int>(&dev_count, 1);
      device_counts.at(i) = dev_count;
    }

    int nccl_rank = 0;
    int nccl_rank_offset = std::accumulate(device_counts.begin(),
                             device_counts.begin() + rabit::GetRank(), 0);
    int nccl_nranks = std::accumulate(device_counts.begin(),
                        device_counts.end(), 0);
    nccl_rank += nccl_rank_offset;

    GroupStart();
    for (size_t i = 0; i < device_ordinals.size(); i++) {
      int dev = device_ordinals.at(i);
      dh::safe_cuda(cudaSetDevice(dev));
      dh::safe_nccl(ncclCommInitRank(
        &comms.at(i),
        nccl_nranks, id,
        nccl_rank));

      nccl_rank++;
    }
    GroupEnd();

    for (size_t i = 0; i < device_ordinals.size(); i++) {
      safe_cuda(cudaSetDevice(device_ordinals.at(i)));
      safe_cuda(cudaStreamCreate(&streams.at(i)));
    }
    initialised_ = true;
#else
    CHECK_EQ(device_ordinals.size(), 1)
        << "XGBoost must be compiled with NCCL to use more than one GPU.";
#endif
  }
  ~AllReducer() {
#ifdef XGBOOST_USE_NCCL
    if (initialised_) {
      for (auto &stream : streams) {
        dh::safe_cuda(cudaStreamDestroy(stream));
      }
      for (auto &comm : comms) {
        ncclCommDestroy(comm);
      }
    }
    if (xgboost::ConsoleLogger::ShouldLog(xgboost::ConsoleLogger::LV::kDebug)) {
      LOG(CONSOLE) << "======== NCCL Statistics========";
      LOG(CONSOLE) << "AllReduce calls: " << allreduce_calls_;
      LOG(CONSOLE) << "AllReduce total MB communicated: " << allreduce_bytes_/1000000;
    }
#endif
  }

  /**
   * \brief Use in exactly the same way as ncclGroupStart
   */
  void GroupStart() {
#ifdef XGBOOST_USE_NCCL
    dh::safe_nccl(ncclGroupStart());
#endif
  }

  /**
   * \brief Use in exactly the same way as ncclGroupEnd
   */
  void GroupEnd() {
#ifdef XGBOOST_USE_NCCL
    dh::safe_nccl(ncclGroupEnd());
#endif
  }

  /**
   * \brief Allreduce. Use in exactly the same way as NCCL but without needing
   * streams or comms.
   *
   * \param communication_group_idx Zero-based index of the communication group.
   * \param sendbuff                The sendbuff.
   * \param recvbuff                The recvbuff.
   * \param count                   Number of elements.
   */

  void AllReduceSum(int communication_group_idx, const double *sendbuff,
                    double *recvbuff, int count) {
#ifdef XGBOOST_USE_NCCL
    CHECK(initialised_);
    dh::safe_cuda(cudaSetDevice(device_ordinals.at(communication_group_idx)));
    dh::safe_nccl(ncclAllReduce(sendbuff, recvbuff, count, ncclDouble, ncclSum,
                                comms.at(communication_group_idx),
                                streams.at(communication_group_idx)));
    if(communication_group_idx == 0)
    {
      allreduce_bytes_ += count * sizeof(double);
      allreduce_calls_ += 1;
    }
#endif
  }

  /**
   * \brief Allreduce. Use in exactly the same way as NCCL but without needing
   * streams or comms.
   *
   * \param communication_group_idx Zero-based index of the communication group.
   * \param sendbuff                The sendbuff.
   * \param recvbuff                The recvbuff.
   * \param count                   Number of elements.
   */

  void AllReduceSum(int communication_group_idx, const float *sendbuff,
                    float *recvbuff, int count) {
#ifdef XGBOOST_USE_NCCL
    CHECK(initialised_);
    dh::safe_cuda(cudaSetDevice(device_ordinals.at(communication_group_idx)));
    dh::safe_nccl(ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum,
                                comms.at(communication_group_idx),
                                streams.at(communication_group_idx)));
    if(communication_group_idx == 0)
    {
      allreduce_bytes_ += count * sizeof(float);
      allreduce_calls_ += 1;
    }
#endif
  }

  /**
   * \brief Allreduce. Use in exactly the same way as NCCL but without needing streams or comms.
   *
   * \param count Number of.
   *
   * \param communication_group_idx Zero-based index of the communication group. \param sendbuff.
   * \param sendbuff                The sendbuff.
   * \param recvbuff                The recvbuff.
   * \param count                   Number of.
   */

  void AllReduceSum(int communication_group_idx, const int64_t *sendbuff,
                    int64_t *recvbuff, int count) {
#ifdef XGBOOST_USE_NCCL
    CHECK(initialised_);

    dh::safe_cuda(cudaSetDevice(device_ordinals[communication_group_idx]));
    dh::safe_nccl(ncclAllReduce(sendbuff, recvbuff, count, ncclInt64, ncclSum,
                                comms[communication_group_idx],
                                streams[communication_group_idx]));
#endif
  }

  /**
   * \fn  void Synchronize()
   *
   * \brief Synchronizes the entire communication group.
   */
  void Synchronize() {
#ifdef XGBOOST_USE_NCCL
    for (size_t i = 0; i < device_ordinals.size(); i++) {
      dh::safe_cuda(cudaSetDevice(device_ordinals[i]));
      dh::safe_cuda(cudaStreamSynchronize(streams[i]));
    }
#endif
  };

  /**
   * \brief Synchronizes the device
   *
   * \param device_id Identifier for the device.
   */
  void Synchronize(int device_id) {
#ifdef XGBOOST_USE_NCCL
    SaveCudaContext([&]() {
      dh::safe_cuda(cudaSetDevice(device_id));
      int idx = std::find(device_ordinals.begin(), device_ordinals.end(), device_id) - device_ordinals.begin();
      CHECK(idx < device_ordinals.size());
      dh::safe_cuda(cudaStreamSynchronize(streams[idx]));
    });
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
    static const int RootRank = 0;
    ncclUniqueId id;
    if (rabit::GetRank() == RootRank) {
      dh::safe_nccl(ncclGetUniqueId(&id));
    }
    rabit::Broadcast(
      (void*)&id,
      (size_t)sizeof(ncclUniqueId),
      (int)RootRank);
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
    // Wait in case some other thread is accessing host_data
#pragma omp barrier
    // Reset shared buffer
#pragma omp single
    {
      host_data.resize(data.size());
      std::fill(host_data.begin(), host_data.end(), size_t(0));
    }
    // Threads update shared array
    for (auto i = 0ull; i < data.size(); i++) {
#pragma omp critical
      { host_data[i] = std::max(host_data[i], data[i]); }
    }
    // Wait until all threads are finished
#pragma omp barrier

    // One thread performs all reduce across distributed nodes
#pragma omp master
    {
      rabit::Allreduce<rabit::op::Max, size_t>(host_data.data(),
                                               host_data.size());
    }

#pragma omp barrier

    // Threads can now read back all reduced values
    for (auto i = 0ull; i < data.size(); i++) {
      data[i] = host_data[i];
    }
#endif
  }
};

/**
 * \brief Executes some operation on each element of the input vector, using a
 * single controlling thread for each element. In addition, passes the shard index
 * into the function.
 *
 * \tparam  T       Generic type parameter.
 * \tparam  FunctionT  Type of the function t.
 * \param shards  The shards.
 * \param f       The func_t to process.
 */

template <typename T, typename FunctionT>
void ExecuteIndexShards(std::vector<T> *shards, FunctionT f) {
  SaveCudaContext{[&]() {
    // Temporarily turn off dynamic so we have a guaranteed number of threads
    bool dynamic = omp_get_dynamic();
    omp_set_dynamic(false);
    const long shards_size = static_cast<long>(shards->size());
#pragma omp parallel for schedule(static, 1) if (shards_size > 1) num_threads(shards_size)
    for (long shard = 0; shard < shards_size; ++shard) {
      f(shard, shards->at(shard));
    }
    omp_set_dynamic(dynamic);
  }};
}

/**
 * \brief Executes some operation on each element of the input vector, using a single controlling
 *        thread for each element, returns the sum of the results.
 *
 * \tparam  ReduceT  Type of the reduce t.
 * \tparam  T         Generic type parameter.
 * \tparam  FunctionT    Type of the function t.
 * \param shards  The shards.
 * \param f       The func_t to process.
 *
 * \return  A reduce_t.
 */

template <typename ReduceT, typename ShardT, typename FunctionT>
ReduceT ReduceShards(std::vector<ShardT> *shards, FunctionT f) {
  std::vector<ReduceT> sums(shards->size());
  SaveCudaContext {
    [&](){
#pragma omp parallel for schedule(static, 1) if (shards->size() > 1)
      for (int shard = 0; shard < shards->size(); ++shard) {
        sums[shard] = f(shards->at(shard));
      }}
  };
  return std::accumulate(sums.begin(), sums.end(), ReduceT());
}

template <typename T,
  typename IndexT = typename xgboost::common::Span<T>::index_type>
xgboost::common::Span<T> ToSpan(
    device_vector<T>& vec,
    IndexT offset = 0,
    IndexT size = -1) {
  size = size == -1 ? vec.size() : size;
  CHECK_LE(offset + size, vec.size());
  return {vec.data().get() + offset, static_cast<IndexT>(size)};
}

template <typename T>
xgboost::common::Span<T> ToSpan(thrust::device_vector<T>& vec,
                                size_t offset, size_t size) {
  using IndexT = typename xgboost::common::Span<T>::index_type;
  return ToSpan(vec, static_cast<IndexT>(offset), static_cast<IndexT>(size));
}

template <typename FunctionT>
class LauncherItr {
public:
  int idx;
  FunctionT f;
  XGBOOST_DEVICE LauncherItr() : idx(0) {}
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
 using iterator_category = typename thrust::detail::iterator_facade_category<
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

}  // namespace dh
