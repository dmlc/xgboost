/*!
 * Copyright 2017 XGBoost contributors
 */
#pragma once
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
#include <xgboost/logging.h>
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

#define HOST_DEV_INLINE __host__ __device__ __forceinline__
#define DEV_INLINE __device__ __forceinline__

/*
 * Error handling  functions
 */

#define safe_cuda(ans) throw_on_cuda_error((ans), __FILE__, __LINE__)

inline cudaError_t throw_on_cuda_error(cudaError_t code, const char *file,
                                       int line) {
  if (code != cudaSuccess) {
    std::stringstream ss;
    ss << file << "(" << line << ")";
    std::string file_and_line;
    ss >> file_and_line;
    throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
  }

  return code;
}

#ifdef XGBOOST_USE_NCCL
#define safe_nccl(ans) throw_on_nccl_error((ans), __FILE__, __LINE__)

inline ncclResult_t throw_on_nccl_error(ncclResult_t code, const char *file,
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
T *raw(thrust::device_vector<T> &v) {  //  NOLINT
  return raw_pointer_cast(v.data());
}

template <typename T>
const T *raw(const thrust::device_vector<T> &v) {  //  NOLINT
  return raw_pointer_cast(v.data());
}

inline int n_visible_devices() {
  int n_visgpus = 0;

  dh::safe_cuda(cudaGetDeviceCount(&n_visgpus));

  return n_visgpus;
}

inline int n_devices_all(int n_gpus) {
  int n_devices_visible = dh::n_visible_devices();
  int n_devices = n_gpus < 0 ? n_devices_visible : n_gpus;
  return (n_devices);
}
inline int n_devices(int n_gpus, int num_rows) {
  int n_devices = dh::n_devices_all(n_gpus);
  // fix-up device number to be limited by number of rows
  n_devices = n_devices > num_rows ? num_rows : n_devices;
  return (n_devices);
}

// if n_devices=-1, then use all visible devices
inline void synchronize_n_devices(int n_devices, std::vector<int> dList) {
  for (int d_idx = 0; d_idx < n_devices; d_idx++) {
    int device_idx = dList[d_idx];
    safe_cuda(cudaSetDevice(device_idx));
    safe_cuda(cudaDeviceSynchronize());
  }
}
inline void synchronize_all() {
  for (int device_idx = 0; device_idx < n_visible_devices(); device_idx++) {
    safe_cuda(cudaSetDevice(device_idx));
    safe_cuda(cudaDeviceSynchronize());
  }
}

inline std::string device_name(int device_idx) {
  cudaDeviceProp prop;
  dh::safe_cuda(cudaGetDeviceProperties(&prop, device_idx));
  return std::string(prop.name);
}

inline size_t available_memory(int device_idx) {
  size_t device_free = 0;
  size_t device_total = 0;
  safe_cuda(cudaSetDevice(device_idx));
  dh::safe_cuda(cudaMemGetInfo(&device_free, &device_total));
  return device_free;
}

/**
 * \fn  inline int max_shared_memory(int device_idx)
 *
 * \brief Maximum shared memory per block on this device.
 *
 * \param device_idx  Zero-based index of the device.
 */

inline size_t max_shared_memory(int device_idx) {
  cudaDeviceProp prop;
  dh::safe_cuda(cudaGetDeviceProperties(&prop, device_idx));
  return prop.sharedMemPerBlock;
}

// ensure gpu_id is correct, so not dependent upon user knowing details
inline int get_device_idx(int gpu_id) {
  // protect against overrun for gpu_id
  return (std::abs(gpu_id) + 0) % dh::n_visible_devices();
}

inline void check_compute_capability() {
  int n_devices = n_visible_devices();
  for (int d_idx = 0; d_idx < n_devices; ++d_idx) {
    cudaDeviceProp prop;
    safe_cuda(cudaGetDeviceProperties(&prop, d_idx));
    std::ostringstream oss;
    oss << "CUDA Capability Major/Minor version number: " << prop.major << "."
        << prop.minor << " is insufficient.  Need >=3.5";
    int failed = prop.major < 3 || prop.major == 3 && prop.minor < 5;
    if (failed) LOG(WARNING) << oss.str() << " for device: " << d_idx;
  }
}

/*
 * Range iterator
 */

class range {
 public:
  class iterator {
    friend class range;

   public:
    __host__ __device__ int64_t operator*() const { return i_; }
    __host__ __device__ const iterator &operator++() {
      i_ += step_;
      return *this;
    }
    __host__ __device__ iterator operator++(int) {
      iterator copy(*this);
      i_ += step_;
      return copy;
    }

    __host__ __device__ bool operator==(const iterator &other) const {
      return i_ >= other.i_;
    }
    __host__ __device__ bool operator!=(const iterator &other) const {
      return i_ < other.i_;
    }

    __host__ __device__ void step(int s) { step_ = s; }

   protected:
    __host__ __device__ explicit iterator(int64_t start) : i_(start) {}

   public:
    uint64_t i_;
    int step_ = 1;
  };

  __host__ __device__ iterator begin() const { return begin_; }
  __host__ __device__ iterator end() const { return end_; }
  __host__ __device__ range(int64_t begin, int64_t end)
      : begin_(begin), end_(end) {}
  __host__ __device__ void step(int s) { begin_.step(s); }

 private:
  iterator begin_;
  iterator end_;
};

template <typename T>
__device__ range grid_stride_range(T begin, T end) {
  begin += blockDim.x * blockIdx.x + threadIdx.x;
  range r(begin, end);
  r.step(gridDim.x * blockDim.x);
  return r;
}

template <typename T>
__device__ range block_stride_range(T begin, T end) {
  begin += threadIdx.x;
  range r(begin, end);
  r.step(blockDim.x);
  return r;
}

// Threadblock iterates over range, filling with value. Requires all threads in
// block to be active.
template <typename IterT, typename ValueT>
__device__ void block_fill(IterT begin, size_t n, ValueT value) {
  for (auto i : block_stride_range(static_cast<size_t>(0), n)) {
    begin[i] = value;
  }
}

/*
 * Kernel launcher
 */

template <typename T1, typename T2>
T1 div_round_up(const T1 a, const T2 b) {
  return static_cast<T1>(ceil(static_cast<double>(a) / b));
}

template <typename L>
__global__ void launch_n_kernel(size_t begin, size_t end, L lambda) {
  for (auto i : grid_stride_range(begin, end)) {
    lambda(i);
  }
}
template <typename L>
__global__ void launch_n_kernel(int device_idx, size_t begin, size_t end,
                                L lambda) {
  for (auto i : grid_stride_range(begin, end)) {
    lambda(i, device_idx);
  }
}

template <int ITEMS_PER_THREAD = 8, int BLOCK_THREADS = 256, typename L>
inline void launch_n(int device_idx, size_t n, L lambda) {
  if (n == 0) {
    return;
  }

  safe_cuda(cudaSetDevice(device_idx));
  const int GRID_SIZE =
      static_cast<int>(div_round_up(n, ITEMS_PER_THREAD * BLOCK_THREADS));
  launch_n_kernel<<<GRID_SIZE, BLOCK_THREADS>>>(static_cast<size_t>(0), n,
                                                lambda);
}

/*
 * Memory
 */

enum memory_type { DEVICE, DEVICE_MANAGED };

template <memory_type MemoryT>
class bulk_allocator;
template <typename T>
class dvec2;

template <typename T>
class dvec {
  friend class dvec2<T>;

 private:
  T *_ptr;
  size_t _size;
  int _device_idx;

 public:
  void external_allocate(int device_idx, void *ptr, size_t size) {
    if (!empty()) {
      throw std::runtime_error("Tried to allocate dvec but already allocated");
    }
    _ptr = static_cast<T *>(ptr);
    _size = size;
    _device_idx = device_idx;
    safe_cuda(cudaSetDevice(_device_idx));
  }

  dvec() : _ptr(NULL), _size(0), _device_idx(-1) {}
  size_t size() const { return _size; }
  int device_idx() const { return _device_idx; }
  bool empty() const { return _ptr == NULL || _size == 0; }

  T *data() { return _ptr; }

  const T *data() const { return _ptr; }

  std::vector<T> as_vector() const {
    std::vector<T> h_vector(size());
    safe_cuda(cudaSetDevice(_device_idx));
    safe_cuda(cudaMemcpy(h_vector.data(), _ptr, size() * sizeof(T),
                         cudaMemcpyDeviceToHost));
    return h_vector;
  }

  void fill(T value) {
    auto d_ptr = _ptr;
    launch_n(_device_idx, size(),
             [=] __device__(size_t idx) { d_ptr[idx] = value; });
  }

  void print() {
    auto h_vector = this->as_vector();
    for (auto e : h_vector) {
      std::cout << e << " ";
    }
    std::cout << "\n";
  }

  thrust::device_ptr<T> tbegin() { return thrust::device_pointer_cast(_ptr); }

  thrust::device_ptr<T> tend() {
    return thrust::device_pointer_cast(_ptr + size());
  }

  template <typename T2>
  dvec &operator=(const std::vector<T2> &other) {
    this->copy(other.begin(), other.end());
    return *this;
  }

  dvec &operator=(dvec<T> &other) {
    if (other.size() != size()) {
      throw std::runtime_error(
          "Cannot copy assign dvec to dvec, sizes are different");
    }
    safe_cuda(cudaSetDevice(this->device_idx()));
    if (other.device_idx() == this->device_idx()) {
      dh::safe_cuda(cudaMemcpy(this->data(), other.data(),
                               other.size() * sizeof(T),
                               cudaMemcpyDeviceToDevice));
    } else {
      std::cout << "deviceother: " << other.device_idx()
                << " devicethis: " << this->device_idx() << std::endl;
      std::cout << "size deviceother: " << other.size()
                << " devicethis: " << this->device_idx() << std::endl;
      throw std::runtime_error("Cannot copy to/from different devices");
    }

    return *this;
  }

  template <typename IterT>
  void copy(IterT begin, IterT end) {
    safe_cuda(cudaSetDevice(this->device_idx()));
    if (end - begin != size()) {
      throw std::runtime_error(
          "Cannot copy assign vector to dvec, sizes are different");
    }
    thrust::copy(begin, end, this->tbegin());
  }
};

/**
 * @class dvec2 device_helpers.cuh
 * @brief wrapper for storing 2 dvec's which are needed for cub::DoubleBuffer
 */
template <typename T>
class dvec2 {
 private:
  dvec<T> _d1, _d2;
  cub::DoubleBuffer<T> _buff;
  int _device_idx;

 public:
  void external_allocate(int device_idx, void *ptr1, void *ptr2, size_t size) {
    if (!empty()) {
      throw std::runtime_error("Tried to allocate dvec2 but already allocated");
    }
    _device_idx = device_idx;
    _d1.external_allocate(_device_idx, ptr1, size);
    _d2.external_allocate(_device_idx, ptr2, size);
    _buff.d_buffers[0] = static_cast<T *>(ptr1);
    _buff.d_buffers[1] = static_cast<T *>(ptr2);
    _buff.selector = 0;
  }
  dvec2() : _d1(), _d2(), _buff(), _device_idx(-1) {}

  size_t size() const { return _d1.size(); }
  int device_idx() const { return _device_idx; }
  bool empty() const { return _d1.empty() || _d2.empty(); }

  cub::DoubleBuffer<T> &buff() { return _buff; }

  dvec<T> &d1() { return _d1; }
  dvec<T> &d2() { return _d2; }

  T *current() { return _buff.Current(); }

  dvec<T> &current_dvec() { return _buff.selector == 0 ? d1() : d2(); }

  T *other() { return _buff.Alternate(); }
};

template <memory_type MemoryT>
class bulk_allocator {
  std::vector<char *> d_ptr;
  std::vector<size_t> _size;
  std::vector<int> _device_idx;

  const int align = 256;

  size_t align_round_up(size_t n) const {
    n = (n + align - 1) / align;
    return n * align;
  }

  template <typename T>
  size_t get_size_bytes(dvec<T> *first_vec, size_t first_size) {
    return align_round_up(first_size * sizeof(T));
  }

  template <typename T, typename... Args>
  size_t get_size_bytes(dvec<T> *first_vec, size_t first_size, Args... args) {
    return get_size_bytes<T>(first_vec, first_size) + get_size_bytes(args...);
  }

  template <typename T>
  void allocate_dvec(int device_idx, char *ptr, dvec<T> *first_vec,
                     size_t first_size) {
    first_vec->external_allocate(device_idx, static_cast<void *>(ptr),
                                 first_size);
  }

  template <typename T, typename... Args>
  void allocate_dvec(int device_idx, char *ptr, dvec<T> *first_vec,
                     size_t first_size, Args... args) {
    allocate_dvec<T>(device_idx, ptr, first_vec, first_size);
    ptr += align_round_up(first_size * sizeof(T));
    allocate_dvec(device_idx, ptr, args...);
  }

  char *allocate_device(int device_idx, size_t bytes, memory_type t) {
    char *ptr;
    safe_cuda(cudaSetDevice(device_idx));
    safe_cuda(cudaMalloc(&ptr, bytes));
    return ptr;
  }
  template <typename T>
  size_t get_size_bytes(dvec2<T> *first_vec, size_t first_size) {
    return 2 * align_round_up(first_size * sizeof(T));
  }

  template <typename T, typename... Args>
  size_t get_size_bytes(dvec2<T> *first_vec, size_t first_size, Args... args) {
    return get_size_bytes<T>(first_vec, first_size) + get_size_bytes(args...);
  }

  template <typename T>
  void allocate_dvec(int device_idx, char *ptr, dvec2<T> *first_vec,
                     size_t first_size) {
    first_vec->external_allocate(
        device_idx, static_cast<void *>(ptr),
        static_cast<void *>(ptr + align_round_up(first_size * sizeof(T))),
        first_size);
  }

  template <typename T, typename... Args>
  void allocate_dvec(int device_idx, char *ptr, dvec2<T> *first_vec,
                     size_t first_size, Args... args) {
    allocate_dvec<T>(device_idx, ptr, first_vec, first_size);
    ptr += (align_round_up(first_size * sizeof(T)) * 2);
    allocate_dvec(device_idx, ptr, args...);
  }

 public:
  ~bulk_allocator() {
    for (size_t i = 0; i < d_ptr.size(); i++) {
      if (!(d_ptr[i] == nullptr)) {
        safe_cuda(cudaSetDevice(_device_idx[i]));
        safe_cuda(cudaFree(d_ptr[i]));
        d_ptr[i] = nullptr;
      }
    }
  }

  // returns sum of bytes for all allocations
  size_t size() {
    return std::accumulate(_size.begin(), _size.end(), static_cast<size_t>(0));
  }

  template <typename... Args>
  void allocate(int device_idx, bool silent, Args... args) {
    size_t size = get_size_bytes(args...);

    char *ptr = allocate_device(device_idx, size, MemoryT);

    allocate_dvec(device_idx, ptr, args...);

    d_ptr.push_back(ptr);
    _size.push_back(size);
    _device_idx.push_back(device_idx);

    if (!silent) {
      const int mb_size = 1048576;
      LOG(CONSOLE) << "Allocated " << size / mb_size << "MB on [" << device_idx
                   << "] " << device_name(device_idx) << ", "
                   << available_memory(device_idx) / mb_size << "MB remaining.";
    }
  }
};

// Keep track of cub library device allocation
struct CubMemory {
  void *d_temp_storage;
  size_t temp_storage_bytes;

  // Thrust
  typedef char value_type;

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
  char *allocate(std::ptrdiff_t num_bytes) {
    LazyAllocate(num_bytes);
    return reinterpret_cast<char *>(d_temp_storage);
  }

  // Thrust
  void deallocate(char *ptr, size_t n) {
    // Do nothing
  }

  bool IsAllocated() { return d_temp_storage != NULL; }
};

/*
 *  Utility functions
 */

template <typename T>
void print(const dvec<T> &v, size_t max_items = 10) {
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

template <typename coordinate_t, typename segments_t, typename offset_t>
void FindMergePartitions(int device_idx, coordinate_t *d_tile_coordinates,
                         size_t num_tiles, int tile_size, segments_t segments,
                         offset_t num_rows, offset_t num_elements) {
  dh::launch_n(device_idx, num_tiles + 1, [=] __device__(int idx) {
    offset_t diagonal = idx * tile_size;
    coordinate_t tile_coordinate;
    cub::CountingInputIterator<offset_t> nonzero_indices(0);

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
          typename offset_t, typename coordinate_t, typename func_t,
          typename segments_iter>
__global__ void LbsKernel(coordinate_t *d_coordinates,
                          segments_iter segment_end_offsets, func_t f,
                          offset_t num_segments) {
  int tile = blockIdx.x;
  coordinate_t tile_start_coord = d_coordinates[tile];
  coordinate_t tile_end_coord = d_coordinates[tile + 1];
  int64_t tile_num_rows = tile_end_coord.x - tile_start_coord.x;
  int64_t tile_num_elements = tile_end_coord.y - tile_start_coord.y;

  cub::CountingInputIterator<offset_t> tile_element_indices(tile_start_coord.y);
  coordinate_t thread_start_coord;

  typedef typename std::iterator_traits<segments_iter>::value_type segment_t;
  __shared__ struct {
    segment_t tile_segment_end_offsets[TILE_SIZE + 1];
    segment_t output_segment[TILE_SIZE];
  } temp_storage;

  for (auto item : dh::block_stride_range(int(0), int(tile_num_rows + 1))) {
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

  coordinate_t thread_current_coord = thread_start_coord;
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

  for (auto item : dh::block_stride_range(int(0), int(tile_num_elements))) {
    f(tile_start_coord.y + item, temp_storage.output_segment[item]);
  }
}

template <typename func_t, typename segments_iter, typename offset_t>
void SparseTransformLbs(int device_idx, dh::CubMemory *temp_memory,
                        offset_t count, segments_iter segments,
                        offset_t num_segments, func_t f) {
  typedef typename cub::CubVector<offset_t, 2>::Type coordinate_t;
  dh::safe_cuda(cudaSetDevice(device_idx));
  const int BLOCK_THREADS = 256;
  const int ITEMS_PER_THREAD = 1;
  const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
  auto num_tiles = dh::div_round_up(count + num_segments, BLOCK_THREADS);
  CHECK(num_tiles < std::numeric_limits<unsigned int>::max());

  temp_memory->LazyAllocate(sizeof(coordinate_t) * (num_tiles + 1));
  coordinate_t *tmp_tile_coordinates =
      reinterpret_cast<coordinate_t *>(temp_memory->d_temp_storage);

  FindMergePartitions(device_idx, tmp_tile_coordinates, num_tiles,
                      BLOCK_THREADS, segments, num_segments, count);

  LbsKernel<TILE_SIZE, ITEMS_PER_THREAD, BLOCK_THREADS, offset_t>
      <<<uint32_t(num_tiles), BLOCK_THREADS>>>(tmp_tile_coordinates,
                                               segments + 1, f, num_segments);
}

template <typename func_t, typename offset_t>
void DenseTransformLbs(int device_idx, offset_t count, offset_t num_segments,
                       func_t f) {
  CHECK(count % num_segments == 0) << "Data is not dense.";

  launch_n(device_idx, count, [=] __device__(offset_t idx) {
    offset_t segment = idx / (count / num_segments);
    f(idx, segment);
  });
}

/**
 * \fn  template <typename func_t, typename segments_iter, typename offset_t>
 * void TransformLbs(int device_idx, dh::CubMemory *temp_memory, offset_t count,
 * segments_iter segments, offset_t num_segments, bool is_dense, func_t f)
 *
 * \brief Load balancing search function. Reads a CSR type matrix description
 * and allows a function to be executed on each element. Search 'modern GPU load
 * balancing search' for more information.
 *
 * \author  Rory
 * \date  7/9/2017
 *
 * \tparam  func_t        Type of the function t.
 * \tparam  segments_iter Type of the segments iterator.
 * \tparam  offset_t      Type of the offset.
 * \param           device_idx    Zero-based index of the device.
 * \param [in,out]  temp_memory   Temporary memory allocator.
 * \param           count         Number of elements.
 * \param           segments      Device pointer to segments.
 * \param           num_segments  Number of segments.
 * \param           is_dense      True if this object is dense.
 * \param           f             Lambda to be executed on matrix elements.
 */

template <typename func_t, typename segments_iter, typename offset_t>
void TransformLbs(int device_idx, dh::CubMemory *temp_memory, offset_t count,
                  segments_iter segments, offset_t num_segments, bool is_dense,
                  func_t f) {
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
void segmentedSort(dh::CubMemory *tmp_mem, dh::dvec2<T1> *keys,
                   dh::dvec2<T2> *vals, int nVals, int nSegs,
                   const dh::dvec<int> &offsets, int start = 0,
                   int end = sizeof(T1) * 8) {
  size_t tmpSize;
  dh::safe_cuda(cub::DeviceSegmentedRadixSort::SortPairs(
      NULL, tmpSize, keys->buff(), vals->buff(), nVals, nSegs, offsets.data(),
      offsets.data() + 1, start, end));
  tmp_mem->LazyAllocate(tmpSize);
  dh::safe_cuda(cub::DeviceSegmentedRadixSort::SortPairs(
      tmp_mem->d_temp_storage, tmpSize, keys->buff(), vals->buff(), nVals,
      nSegs, offsets.data(), offsets.data() + 1, start, end));
}

/**
 * @brief Helper function to perform device-wide sum-reduction
 * @param tmp_mem cub temporary memory info
 * @param in the input array to be reduced
 * @param out the output reduced value
 * @param nVals number of elements in the input array
 */
template <typename T>
void sumReduction(dh::CubMemory &tmp_mem, dh::dvec<T> &in, dh::dvec<T> &out,
                  int nVals) {
  size_t tmpSize;
  dh::safe_cuda(
      cub::DeviceReduce::Sum(NULL, tmpSize, in.data(), out.data(), nVals));
  tmp_mem.LazyAllocate(tmpSize);
  dh::safe_cuda(cub::DeviceReduce::Sum(tmp_mem.d_temp_storage, tmpSize,
                                       in.data(), out.data(), nVals));
}

/**
 * @brief Fill a given constant value across all elements in the buffer
 * @param out the buffer to be filled
 * @param len number of elements i the buffer
 * @param def default value to be filled
 */
template <typename T, int BlkDim = 256, int ItemsPerThread = 4>
void fillConst(int device_idx, T *out, int len, T def) {
  dh::launch_n<ItemsPerThread, BlkDim>(device_idx, len,
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
void gather(int device_idx, T1 *out1, const T1 *in1, T2 *out2, const T2 *in2,
            const int *instId, int nVals) {
  dh::launch_n<ItemsPerThread, BlkDim>(device_idx, nVals,
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
void gather(int device_idx, T *out, const T *in, const int *instId, int nVals) {
  dh::launch_n<ItemsPerThread, BlkDim>(device_idx, nVals,
                                       [=] __device__(int i) {
                                         int iid = instId[i];
                                         out[i] = in[iid];
                                       });
}

/**
 * \class AllReducer
 *
 * \brief All reducer class that manages its own communication group and
 * streams. Must be initialised before use. If XGBoost is compiled without NCCL this is a dummy class that will error if used with more than one GPU.
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
    CHECK_EQ(device_ordinals.size(), 1) << "XGBoost must be compiled with NCCL to use more than one GPU.";
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
   * \fn  void AllReduceSum(int communication_group_idx, const double *sendbuff,
   * double *recvbuff, int count)
   *
   * \brief Allreduce. Use in exactly the same way as NCCL but without needing
   * streams or comms.
   *
   * \param           communication_group_idx Zero-based index of the
   * communication group. \param sendbuff                The sendbuff. \param
   * sendbuff                The sendbuff. \param [in,out]  recvbuff
   * The recvbuff. \param           count                   Number of.
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
   * \fn  void AllReduceSum(int communication_group_idx, const int64_t *sendbuff, int64_t *recvbuff, int count)
   *
   * \brief Allreduce. Use in exactly the same way as NCCL but without needing streams or comms.
   *
   * \param           communication_group_idx Zero-based index of the communication group. \param
   *                                          sendbuff                The sendbuff. \param sendbuff
   *                                          The sendbuff. \param [in,out]  recvbuff The recvbuff.
   *                                          \param           count                   Number of.
   * \param           sendbuff                The sendbuff.
   * \param [in,out]  recvbuff                If non-null, the recvbuff.
   * \param           count                   Number of.
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
}  // namespace dh
