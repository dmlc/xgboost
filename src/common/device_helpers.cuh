/*!
 * Copyright 2017 XGBoost contributors
 */
#pragma once
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
#include <xgboost/logging.h>

#include "common.h"
#include "gpu_set.h"
#include "span.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <cub/cub.cuh>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#ifdef XGBOOST_USE_NCCL
#include "nccl.h"
#endif

// Uncomment to enable
#define TIMERS

namespace dh {

/*! \brief Threads per block based on heuristic. */
constexpr size_t kBlockThreads = 256;

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

template <typename T>
T *Raw(thrust::device_vector<T> &v) {  //  NOLINT
  return raw_pointer_cast(v.data());
}

template <typename T>
const T *Raw(const thrust::device_vector<T> &v) {  //  NOLINT
  return raw_pointer_cast(v.data());
}

// if n_devices=-1, then use all visible devices
inline void SynchronizeNDevices(xgboost::GPUSet devices) {
  devices = devices.IsEmpty() ? xgboost::GPUSet::AllVisible() : devices;
  for (auto const d : devices.Unnormalised()) {
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
  if (n == 0) {
    return 0;
	}
  if (cuts[n - 1] <= v) {
    return n;
	}
  if (cuts[0] > v) {
    return 0;
	}
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

template <typename T1, typename T2>
T1 DivRoundUp(const T1 a, const T2 b) {
  return static_cast<T1>(ceil(static_cast<double>(a) / b));
}

inline void RowSegments(size_t n_rows, size_t n_devices, std::vector<size_t>* segments) {
  segments->push_back(0);
  size_t row_begin = 0;
  size_t shard_size = DivRoundUp(n_rows, n_devices);
  for (size_t d_idx = 0; d_idx < n_devices; ++d_idx) {
    size_t row_end = std::min(row_begin + shard_size, n_rows);
    segments->push_back(row_end);
    row_begin = row_end;
  }
}


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
inline void LaunchN(int device_idx, size_t n, L lambda) {
  if (n == 0) {
    return;
  }

  safe_cuda(cudaSetDevice(device_idx));
  const int GRID_SIZE =
      static_cast<int>(DivRoundUp(n, ITEMS_PER_THREAD * BLOCK_THREADS));
  LaunchNKernel<<<GRID_SIZE, BLOCK_THREADS>>>(static_cast<size_t>(0), n,
                                                lambda);
}

/*
 * Memory
 */

enum MemoryType { kDevice, kDeviceManaged };

template <MemoryType MemoryT>
class BulkAllocator;
template <typename T>
class DVec2;

template <typename T>
class DVec {
  friend class DVec2<T>;

 private:
  T *ptr_;
  size_t size_;
  int device_idx_;

 public:
  void ExternalAllocate(int device_idx, void *ptr, size_t size) {
    if (!Empty()) {
      throw std::runtime_error("Tried to allocate DVec but already allocated");
    }
    ptr_ = static_cast<T *>(ptr);
    size_ = size;
    device_idx_ = device_idx;
    safe_cuda(cudaSetDevice(device_idx_));
  }

  DVec() : ptr_(NULL), size_(0), device_idx_(-1) {}
  size_t Size() const { return size_; }
  int DeviceIdx() const { return device_idx_; }
  bool Empty() const { return ptr_ == NULL || size_ == 0; }

  T *Data() { return ptr_; }

  const T *Data() const { return ptr_; }

  std::vector<T> AsVector() const {
    std::vector<T> h_vector(Size());
    safe_cuda(cudaSetDevice(device_idx_));
    safe_cuda(cudaMemcpy(h_vector.data(), ptr_, Size() * sizeof(T),
                         cudaMemcpyDeviceToHost));
    return h_vector;
  }

  void Fill(T value) {
    auto d_ptr = ptr_;
    LaunchN(device_idx_, Size(),
             [=] __device__(size_t idx) { d_ptr[idx] = value; });
  }

  void Print() {
    auto h_vector = this->AsVector();
    for (auto e : h_vector) {
      std::cout << e << " ";
    }
    std::cout << "\n";
  }

  thrust::device_ptr<T> tbegin() { return thrust::device_pointer_cast(ptr_); }

  thrust::device_ptr<T> tend() {
    return thrust::device_pointer_cast(ptr_ + Size());
  }

  template <typename T2>
  DVec &operator=(const std::vector<T2> &other) {
    this->copy(other.begin(), other.end());
    return *this;
  }

  DVec &operator=(DVec<T> &other) {
    if (other.Size() != Size()) {
      throw std::runtime_error(
          "Cannot copy assign DVec to DVec, sizes are different");
    }
    safe_cuda(cudaSetDevice(this->DeviceIdx()));
    if (other.DeviceIdx() == this->DeviceIdx()) {
      dh::safe_cuda(cudaMemcpy(this->Data(), other.Data(),
                               other.Size() * sizeof(T),
                               cudaMemcpyDeviceToDevice));
    } else {
      std::cout << "deviceother: " << other.DeviceIdx()
                << " devicethis: " << this->DeviceIdx() << std::endl;
      std::cout << "size deviceother: " << other.Size()
                << " devicethis: " << this->DeviceIdx() << std::endl;
      throw std::runtime_error("Cannot copy to/from different devices");
    }

    return *this;
  }

  template <typename IterT>
  void copy(IterT begin, IterT end) {
    safe_cuda(cudaSetDevice(this->DeviceIdx()));
    if (end - begin != Size()) {
      throw std::runtime_error(
          "Cannot copy assign vector to DVec, sizes are different");
    }
    thrust::copy(begin, end, this->tbegin());
  }

  void copy(thrust::device_ptr<T> begin, thrust::device_ptr<T> end) {
    safe_cuda(cudaSetDevice(this->DeviceIdx()));
    if (end - begin != Size()) {
      throw std::runtime_error(
          "Cannot copy assign vector to dvec, sizes are different");
    }
    safe_cuda(cudaMemcpy(this->Data(), begin.get(), Size() * sizeof(T),
                         cudaMemcpyDefault));
  }
};

/**
 * @class DVec2 device_helpers.cuh
 * @brief wrapper for storing 2 DVec's which are needed for cub::DoubleBuffer
 */
template <typename T>
class DVec2 {
 private:
  DVec<T> d1_, d2_;
  cub::DoubleBuffer<T> buff_;
  int device_idx_;

 public:
  void ExternalAllocate(int device_idx, void *ptr1, void *ptr2, size_t size) {
    if (!Empty()) {
      throw std::runtime_error("Tried to allocate DVec2 but already allocated");
    }
    device_idx_ = device_idx;
    d1_.ExternalAllocate(device_idx_, ptr1, size);
    d2_.ExternalAllocate(device_idx_, ptr2, size);
    buff_.d_buffers[0] = static_cast<T *>(ptr1);
    buff_.d_buffers[1] = static_cast<T *>(ptr2);
    buff_.selector = 0;
  }
  DVec2() : d1_(), d2_(), buff_(), device_idx_(-1) {}

  size_t Size() const { return d1_.Size(); }
  int DeviceIdx() const { return device_idx_; }
  bool Empty() const { return d1_.Empty() || d2_.Empty(); }

  cub::DoubleBuffer<T> &buff() { return buff_; }

  DVec<T> &D1() { return d1_; }

  DVec<T> &D2() { return d2_; }

  T *Current() { return buff_.Current(); }

  DVec<T> &CurrentDVec() { return buff_.selector == 0 ? D1() : D2(); }

  T *other() { return buff_.Alternate(); }
};

template <MemoryType MemoryT>
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
  size_t GetSizeBytes(DVec<T> *first_vec, size_t first_size) {
    return AlignRoundUp(first_size * sizeof(T));
  }

  template <typename T, typename... Args>
  size_t GetSizeBytes(DVec<T> *first_vec, size_t first_size, Args... args) {
    return GetSizeBytes<T>(first_vec, first_size) + GetSizeBytes(args...);
  }

  template <typename T>
  void AllocateDVec(int device_idx, char *ptr, DVec<T> *first_vec,
                     size_t first_size) {
    first_vec->ExternalAllocate(device_idx, static_cast<void *>(ptr),
                                 first_size);
  }

  template <typename T, typename... Args>
  void AllocateDVec(int device_idx, char *ptr, DVec<T> *first_vec,
                     size_t first_size, Args... args) {
    AllocateDVec<T>(device_idx, ptr, first_vec, first_size);
    ptr += AlignRoundUp(first_size * sizeof(T));
    AllocateDVec(device_idx, ptr, args...);
  }

  char *AllocateDevice(int device_idx, size_t bytes) {
    char *ptr;
    safe_cuda(cudaSetDevice(device_idx));
    safe_cuda(cudaMalloc(&ptr, bytes));
    return ptr;
  }
  template <typename T>
  size_t GetSizeBytes(DVec2<T> *first_vec, size_t first_size) {
    return 2 * AlignRoundUp(first_size * sizeof(T));
  }

  template <typename T, typename... Args>
  size_t GetSizeBytes(DVec2<T> *first_vec, size_t first_size, Args... args) {
    return GetSizeBytes<T>(first_vec, first_size) + GetSizeBytes(args...);
  }

  template <typename T>
  void AllocateDVec(int device_idx, char *ptr, DVec2<T> *first_vec,
                    size_t first_size) {
    first_vec->ExternalAllocate(
        device_idx, static_cast<void *>(ptr),
        static_cast<void *>(ptr + AlignRoundUp(first_size * sizeof(T))),
        first_size);
  }

  template <typename T, typename... Args>
  void AllocateDVec(int device_idx, char *ptr, DVec2<T> *first_vec,
                     size_t first_size, Args... args) {
    AllocateDVec<T>(device_idx, ptr, first_vec, first_size);
    ptr += (AlignRoundUp(first_size * sizeof(T)) * 2);
    AllocateDVec(device_idx, ptr, args...);
  }

 public:
  BulkAllocator() = default;
  // prevent accidental copying, moving or assignment of this object
  BulkAllocator(const BulkAllocator<MemoryT>&) = delete;
  BulkAllocator(BulkAllocator<MemoryT>&&) = delete;
  void operator=(const BulkAllocator<MemoryT>&) = delete;
  void operator=(BulkAllocator<MemoryT>&&) = delete;

  ~BulkAllocator() {
    for (size_t i = 0; i < d_ptr_.size(); i++) {
      if (!(d_ptr_[i] == nullptr)) {
        safe_cuda(cudaSetDevice(device_idx_[i]));
        safe_cuda(cudaFree(d_ptr_[i]));
        d_ptr_[i] = nullptr;
      }
    }
  }

  // returns sum of bytes for all allocations
  size_t Size() {
    return std::accumulate(size_.begin(), size_.end(), static_cast<size_t>(0));
  }

  template <typename... Args>
  void Allocate(int device_idx, bool silent, Args... args) {
    size_t size = GetSizeBytes(args...);

    char *ptr = AllocateDevice(device_idx, size);

    AllocateDVec(device_idx, ptr, args...);

    d_ptr_.push_back(ptr);
    size_.push_back(size);
    device_idx_.push_back(device_idx);
  }
};

template <typename T>
class DSpan : public xgboost::common::Span<T> {
  int device_idx_;
  using Byte = xgboost::common::byte;

 public:
  using Span = xgboost::common::Span<T>;
  using index_type = typename Span::index_type;

  DSpan() = default;
  DSpan(T * ptr, index_type size, int device) :
      Span{ptr, size}, device_idx_{device} {}
  DSpan(const DSpan& other) :
      device_idx_{other.device_idx_}, Span{other} {}
  DSpan(Span span, int device) : Span{span}, device_idx_{device} {}

  // DSpan& operator=(const std::vector<T>& vec) {
  //   std::copy(vec.cbegin(), vec.cend(), this->begin());
  //   return *this;
  // }

  void Fill(T value) {
    // FIXME: Pass a span?
    T *ptr = Span::data();
    LaunchN(device_idx_, Span::size(),
            [=] __device__(size_t idx) {
              ptr[idx] = value;
            });
  }

  template <typename IterT>
  void copy(IterT begin, IterT end) {
    safe_cuda(cudaSetDevice(this->DeviceIdx()));
    if (end - begin != Span::size()) {
      LOG(FATAL) << "Cannot copy assign vector to DVec, sizes are different";
    }
    thrust::device_ptr<T> ptr {Span::data()};
    thrust::copy(begin, end, ptr);
  }
  void copy(thrust::device_ptr<T> begin, thrust::device_ptr<T> end) {
    safe_cuda(cudaSetDevice(device_idx_));
    CHECK(end - begin != Span::size()) <<
        "Cannot copy assign vector to dvec, sizes are different";
    safe_cuda(cudaMemcpy(this->data(), begin.get(), Span::size() * sizeof(T),
                         cudaMemcpyDefault));
  }

  std::vector<T> AsVector() const {
    std::vector<T> h_vector(Span::size());
    safe_cuda(cudaSetDevice(device_idx_));
    safe_cuda(cudaMemcpy(h_vector.data(), Span::data(), Span::size_bytes(),
                         cudaMemcpyDeviceToHost));
    return h_vector;
  }
  int DeviceIdx() const {
    return device_idx_;
  }
  DSpan first(index_type count) const {
    DSpan<T> res {Span::first(count), device_idx_};
    return res;
  }
  DSpan last(index_type count) const {
    DSpan<T> res {Span::last(count), device_idx_};
    return res;
  }
};

template <typename T>
class DoubleBuffer {
 private:
  DSpan<T> s1_, s2_;
  cub::DoubleBuffer<T> buff_;
  int device_idx_;

 public:
  DoubleBuffer() : buff_{}, device_idx_{-1} {}
  void ExternalAllocate(int device_idx, DSpan<T> s1, DSpan<T> s2) {
    if (!Empty()) {
      LOG(FATAL) << "Tried to allocate DoubleBuffer but already allocated";
    }
    device_idx_ = device_idx;
    this->s1_ = s1;
    this->s2_ = s2;
    buff_.d_buffers[0] = s1.data();
    buff_.d_buffers[1] = s2.data();
    buff_.selector = 0;
  }
  size_t Size() const { return s1_.size(); }
  int DeviceIdx() const { return device_idx_; }
  bool Empty() const { return s1_.size() == 0 || s2_.size() == 0; }

  cub::DoubleBuffer<T> &buff() { return buff_; }

  DSpan<T> &S1() { return s1_; }
  DSpan<T> &S2() { return s2_; }

  T *Current() { return buff_.Current(); }

  DSpan<T> &CurrentDSpan() {
    return buff_.selector == 0 ? S1() : S2();
  }

  T *Other() { return buff_.Alternate(); }
};

class BulkAllocatorTemp {
  using Byte = xgboost::common::byte;
  using index_type = xgboost::common::detail::ptrdiff_t;

  std::vector<Byte *> d_ptr_;
  std::vector<index_type> size_;
  std::vector<int> device_idx_;

  template <typename T>
  DSpan<T> SpanFromByte(DSpan<Byte> span) const {
    DSpan<T> res {
      {reinterpret_cast<T*>(span.data()),
            static_cast<index_type>(span.size() / sizeof(T))},
          span.DeviceIdx()};
    return res;
  }

  template <typename T>
  void AllocateSpan(int device_idx, DSpan<Byte> buffer,
                    DSpan<T> *span, index_type size) {
    *span = SpanFromByte<T>(buffer.first(size * sizeof(T)));
  }
  template <typename Head, typename... Args>
  void AllocateSpan(int device_idx, DSpan<Byte> buffer,
                    DSpan<Head> *span, index_type size,
                    Args... args) {
    AllocateSpan<Head>(device_idx, buffer, span, size);
    *span = SpanFromByte<Head>(buffer.first(size * sizeof(Head)));
    AllocateSpan(device_idx, buffer.last(buffer.size() - size * sizeof(Head)),
                 args...);
  }

  template <typename T>
  DSpan<Byte> AllocateSpan(
      int device_idx, DSpan<Byte> mem_pool,
      DoubleBuffer<T> *buf, index_type size) {
    auto first = SpanFromByte<T>(mem_pool.first(size * sizeof(T)));
    mem_pool = mem_pool.last(mem_pool.size() - size * sizeof(T));
    auto second = SpanFromByte<T>(mem_pool.first(size * sizeof(T)));
    mem_pool = mem_pool.last(mem_pool.size() - size * sizeof(T));
    buf->ExternalAllocate(device_idx, first, second);
    return mem_pool;
  }
  template <typename Head, typename... Args>
  void AllocateSpan(int device_idx, DSpan<Byte> mem_pool,
                    DoubleBuffer<Head> *buf, index_type size, Args... args) {
    mem_pool = AllocateSpan<Head>(device_idx, mem_pool, buf, size);
    AllocateSpan(device_idx, mem_pool, args...);
  }

  template <typename T>
  index_type GetSizeBytes(DoubleBuffer<T> *buf, index_type size) {
    return size * sizeof(T) * 2;
  }
  template <typename Head, typename... Args>
  index_type GetSizeBytes(DoubleBuffer<Head> *head, index_type size,
                          Args... args) {
    return size * sizeof(Head) * 2+ GetSizeBytes(args...);
  }

  template <typename T>
  index_type GetSizeBytes(DSpan<T> *span, index_type size) {
    return size * sizeof(T);
  }
  template <typename Head, typename... Args>
  index_type GetSizeBytes(DSpan<Head> *head, index_type size,
                          Args... spans) {
    return size * sizeof(Head) + GetSizeBytes(spans...);
  }

  Byte *AllocateDevice(int device_idx, size_t size) {
    Byte *ptr;
    safe_cuda(cudaSetDevice(device_idx));
    safe_cuda(cudaMalloc(&ptr, size));
    return ptr;
  }

 public:
  template <typename... Args>
  void Allocate(int device_idx, Args... args) {
    index_type size_in_byte = GetSizeBytes(args...);
    Byte *ptr = AllocateDevice(device_idx, size_in_byte);

    DSpan<Byte> mem_pool {ptr, size_in_byte, device_idx};
    AllocateSpan(device_idx, mem_pool, args...);

    d_ptr_.push_back(ptr);
    size_.push_back(size_in_byte);
    device_idx_.push_back(device_idx);
  }
  ~BulkAllocatorTemp() {
    for (size_t i = 0; i < d_ptr_.size(); i++) {
      if (!d_ptr_[i]) { continue; }
      safe_cuda(cudaSetDevice(device_idx_[i]));
      safe_cuda(cudaFree(d_ptr_[i]));
      d_ptr_[i] = nullptr;
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
  T *Pointer() {
    return static_cast<T *>(d_temp_storage);
  }

  void Free() {
    if (this->IsAllocated()) {
      safe_cuda(cudaFree(d_temp_storage));
    }
  }

  void LazyAllocate(size_t num_bytes) {
    if (num_bytes > temp_storage_bytes) {
      Free();
      safe_cuda(cudaMalloc(&d_temp_storage, num_bytes));
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

template <typename T>
void Print(const DVec<T> &v, size_t max_items = 10) {
  std::vector<T> h = v.as_vector();
  for (size_t i = 0; i < std::min(max_items, h.size()); i++) {
    std::cout << " " << h[i];
  }
  std::cout << "\n";
}

/**
 * @brief Helper macro to measure timing on GPU
 * @param call the GPU call
 * @param name name used to track later
 * @param stream cuda stream where to measure time
 */
#define TIMEIT(call, name)    \
  do {                        \
    dh::Timer t1234;          \
    call;                     \
    t1234.printElapsed(name); \
  } while (0)

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
  auto num_tiles = dh::DivRoundUp(count + num_segments, BLOCK_THREADS);
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
void SegmentedSort(dh::CubMemory *tmp_mem, dh::DVec2<T1> *keys,
                   dh::DVec2<T2> *vals, int nVals, int nSegs,
                   const dh::DVec<int> &offsets, int start = 0,
                   int end = sizeof(T1) * 8) {
  size_t tmpSize;
  dh::safe_cuda(cub::DeviceSegmentedRadixSort::SortPairs(
      NULL, tmpSize, keys->buff(), vals->buff(), nVals, nSegs, offsets.Data(),
      offsets.Data() + 1, start, end));
  tmp_mem->LazyAllocate(tmpSize);
  dh::safe_cuda(cub::DeviceSegmentedRadixSort::SortPairs(
      tmp_mem->d_temp_storage, tmpSize, keys->buff(), vals->buff(), nVals,
      nSegs, offsets.Data(), offsets.Data() + 1, start, end));
}

/**
 * @brief Helper function to perform device-wide sum-reduction
 * @param tmp_mem cub temporary memory info
 * @param in the input array to be reduced
 * @param out the output reduced value
 * @param nVals number of elements in the input array
 */
template <typename T>
void SumReduction(dh::CubMemory &tmp_mem, dh::DVec<T> &in, dh::DVec<T> &out,
                  int nVals) {
  size_t tmpSize;
  dh::safe_cuda(
      cub::DeviceReduce::Sum(NULL, tmpSize, in.Data(), out.Data(), nVals));
  tmp_mem.LazyAllocate(tmpSize);
  dh::safe_cuda(cub::DeviceReduce::Sum(tmp_mem.d_temp_storage, tmpSize,
                                       in.Data(), out.Data(), nVals));
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
  size_t tmpSize;
  dh::safe_cuda(cub::DeviceReduce::Sum(nullptr, tmpSize, in, in, nVals));
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

/**
 * \class AllReducer
 *
 * \brief All reducer class that manages its own communication group and
 * streams. Must be initialised before use. If XGBoost is compiled without NCCL
 * this is a dummy class that will error if used with more than one GPU.
 */

class AllReducer {
  bool initialised;
#ifdef XGBOOST_USE_NCCL
  std::vector<ncclComm_t> comms;
  std::vector<cudaStream_t> streams;
  std::vector<int> device_ordinals;
#endif
 public:
  AllReducer() : initialised(false) {}

  /**
   * \fn  void Init(const std::vector<int> &device_ordinals)
   *
   * \brief Initialise with the desired device ordinals for this communication
   * group.
   *
   * \param device_ordinals The device ordinals.
   */

  void Init(const std::vector<int> &device_ordinals) {
#ifdef XGBOOST_USE_NCCL
    this->device_ordinals = device_ordinals;
    comms.resize(device_ordinals.size());
    dh::safe_nccl(ncclCommInitAll(comms.data(),
                                  static_cast<int>(device_ordinals.size()),
                                  device_ordinals.data()));
    streams.resize(device_ordinals.size());
    for (size_t i = 0; i < device_ordinals.size(); i++) {
      safe_cuda(cudaSetDevice(device_ordinals[i]));
      safe_cuda(cudaStreamCreate(&streams[i]));
    }
    initialised = true;
#else
    CHECK_EQ(device_ordinals.size(), 1)
        << "XGBoost must be compiled with NCCL to use more than one GPU.";
#endif
  }
  ~AllReducer() {
#ifdef XGBOOST_USE_NCCL
    if (initialised) {
      for (auto &stream : streams) {
        dh::safe_cuda(cudaStreamDestroy(stream));
      }
      for (auto &comm : comms) {
        ncclCommDestroy(comm);
      }
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
    CHECK(initialised);

    dh::safe_cuda(cudaSetDevice(device_ordinals[communication_group_idx]));
    dh::safe_nccl(ncclAllReduce(sendbuff, recvbuff, count, ncclDouble, ncclSum,
                                comms[communication_group_idx],
                                streams[communication_group_idx]));
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
    CHECK(initialised);

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
    for (int i = 0; i < device_ordinals.size(); i++) {
      dh::safe_cuda(cudaSetDevice(device_ordinals[i]));
      dh::safe_cuda(cudaStreamSynchronize(streams[i]));
    }
#endif
  }
};

/**
 * \brief Executes some operation on each element of the input vector, using a
 * single controlling thread for each element.
 *
 * \tparam  T       Generic type parameter.
 * \tparam  FunctionT  Type of the function t.
 * \param shards  The shards.
 * \param f       The func_t to process.
 */

template <typename T, typename FunctionT>
void ExecuteShards(std::vector<T> *shards, FunctionT f) {
#pragma omp parallel for schedule(static, 1) if (shards->size() > 1)
  for (int shard = 0; shard < shards->size(); ++shard) {
    f(shards->at(shard));
  }
}

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
#pragma omp parallel for schedule(static, 1) if (shards->size() > 1)
  for (int shard = 0; shard < shards->size(); ++shard) {
    f(shard, shards->at(shard));
  }
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

template <typename ReduceT,typename T, typename FunctionT>
ReduceT ReduceShards(std::vector<T> *shards, FunctionT f) {
  std::vector<ReduceT> sums(shards->size());
#pragma omp parallel for schedule(static, 1) if (shards->size() > 1)
  for (int shard = 0; shard < shards->size(); ++shard) {
    sums[shard] = f(shards->at(shard));
  }
  return std::accumulate(sums.begin(), sums.end(), ReduceT());
}
}  // namespace dh
