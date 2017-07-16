/*!
 * Copyright 2017 XGBoost contributors
 */
#pragma once
#include <dmlc/logging.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system_error.h>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <cub/cub.cuh>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include "nccl.h"

// Uncomment to enable
// #define DEVICE_TIMER
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

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

inline int n_visible_devices() {
  int n_visgpus = 0;

  cudaGetDeviceCount(&n_visgpus);

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

// ensure gpu_id is correct, so not dependent upon user knowing details
inline int get_device_idx(int gpu_id) {
  // protect against overrun for gpu_id
  return (std::abs(gpu_id) + 0) % dh::n_visible_devices();
}

/*
 *  Timers
 */

struct Timer {
  typedef std::chrono::high_resolution_clock ClockT;

  typedef std::chrono::high_resolution_clock::time_point TimePointT;
  TimePointT start;
  Timer() { reset(); }

  void reset() { start = ClockT::now(); }
  int64_t elapsed() const { return (ClockT::now() - start).count(); }
  double elapsedSeconds() const {
    return elapsed() * ((double)ClockT::period::num / ClockT::period::den);
  }
  void printElapsed(std::string label) {
    //    synchronize_n_devices(n_devices, dList);
    printf("%s:\t %fs\n", label.c_str(), elapsedSeconds());
    reset();
  }
};

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

// Threadblock iterates over range, filling with value
template <typename IterT, typename ValueT>
__device__ void block_fill(IterT begin, size_t n, ValueT value) {
  for (auto i : block_stride_range(static_cast<size_t>(0), n)) {
    begin[i] = value;
  }
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
    safe_cuda(cudaSetDevice(_device_idx));
    thrust::fill_n(thrust::device_pointer_cast(_ptr), size(), value);
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
      thrust::copy(other.tbegin(), other.tend(), this->tbegin());
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

  template <typename SizeT>
  size_t align_round_up(SizeT n) {
    n = (n + align - 1) / align;
    return n * align;
  }

  template <typename T, typename SizeT>
  size_t get_size_bytes(dvec<T> *first_vec, SizeT first_size) {
    return align_round_up<SizeT>(first_size * sizeof(T));
  }

  template <typename T, typename SizeT, typename... Args>
  size_t get_size_bytes(dvec<T> *first_vec, SizeT first_size, Args... args) {
    return get_size_bytes<T, SizeT>(first_vec, first_size) +
           get_size_bytes(args...);
  }

  template <typename T, typename SizeT>
  void allocate_dvec(int device_idx, char *ptr, dvec<T> *first_vec,
                     SizeT first_size) {
    first_vec->external_allocate(device_idx, static_cast<void *>(ptr),
                                 first_size);
  }

  template <typename T, typename SizeT, typename... Args>
  void allocate_dvec(int device_idx, char *ptr, dvec<T> *first_vec,
                     SizeT first_size, Args... args) {
    first_vec->external_allocate(device_idx, static_cast<void *>(ptr),
                                 first_size);
    ptr += align_round_up(first_size * sizeof(T));
    allocate_dvec(device_idx, ptr, args...);
  }

  //    template <memory_type MemoryT>
  char *allocate_device(int device_idx, size_t bytes, memory_type t) {
    char *ptr;
    if (t == memory_type::DEVICE) {
      safe_cuda(cudaSetDevice(device_idx));
      safe_cuda(cudaMalloc(&ptr, bytes));
    } else {
      safe_cuda(cudaMallocManaged(&ptr, bytes));
    }
    return ptr;
  }
  template <typename T, typename SizeT>
  size_t get_size_bytes(dvec2<T> *first_vec, SizeT first_size) {
    return 2 * align_round_up(first_size * sizeof(T));
  }

  template <typename T, typename SizeT, typename... Args>
  size_t get_size_bytes(dvec2<T> *first_vec, SizeT first_size, Args... args) {
    return get_size_bytes<T, SizeT>(first_vec, first_size) +
           get_size_bytes(args...);
  }

  template <typename T, typename SizeT>
  void allocate_dvec(int device_idx, char *ptr, dvec2<T> *first_vec,
                     SizeT first_size) {
    first_vec->external_allocate(
        device_idx, static_cast<void *>(ptr),
        static_cast<void *>(ptr + align_round_up(first_size * sizeof(T))),
        first_size);
  }

  template <typename T, typename SizeT, typename... Args>
  void allocate_dvec(int device_idx, char *ptr, dvec2<T> *first_vec,
                     SizeT first_size, Args... args) {
    allocate_dvec<T, SizeT>(device_idx, ptr, first_vec, first_size);
    ptr += (align_round_up(first_size * sizeof(T)) * 2);
    allocate_dvec(device_idx, ptr, args...);
  }

 public:
  ~bulk_allocator() {
    for (int i = 0; i < d_ptr.size(); i++) {
      if (!(d_ptr[i] == nullptr)) {
        safe_cuda(cudaSetDevice(_device_idx[i]));
        safe_cuda(cudaFree(d_ptr[i]));
      }
    }
  }

  // returns sum of bytes for all allocations
  size_t size() {
    return std::accumulate(_size.begin(), _size.end(), static_cast<size_t>(0));
  }

  template <typename... Args>
  void allocate(int device_idx, Args... args) {
    size_t size = get_size_bytes(args...);

    char *ptr = allocate_device(device_idx, size, MemoryT);

    allocate_dvec(device_idx, ptr, args...);

    d_ptr.push_back(ptr);
    _size.push_back(size);
    _device_idx.push_back(device_idx);
  }
};

// Keep track of cub library device allocation
struct CubMemory {
  void *d_temp_storage;
  size_t temp_storage_bytes;

  // Thrust
  typedef char value_type;

  CubMemory() : d_temp_storage(NULL), temp_storage_bytes(0) {}

  ~CubMemory() { Free(); }

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

inline size_t available_memory(int device_idx) {
  size_t device_free = 0;
  size_t device_total = 0;
  safe_cuda(cudaSetDevice(device_idx));
  dh::safe_cuda(cudaMemGetInfo(&device_free, &device_total));
  return device_free;
}

/*
 *  Utility functions
 */

template <typename T>
void print(const thrust::device_vector<T> &v, size_t max_items = 10) {
  thrust::host_vector<T> h = v;
  for (int i = 0; i < std::min(max_items, h.size()); i++) {
    std::cout << " " << h[i];
  }
  std::cout << "\n";
}

template <typename T>
void print(const dvec<T> &v, size_t max_items = 10) {
  std::vector<T> h = v.as_vector();
  for (int i = 0; i < std::min(max_items, h.size()); i++) {
    std::cout << " " << h[i];
  }
  std::cout << "\n";
}

template <typename T>
void print(char *label, const thrust::device_vector<T> &v,
           const char *format = "%d ", int max = 10) {
  thrust::host_vector<T> h_v = v;
  std::cout << label << ":\n";
  for (int i = 0; i < std::min(static_cast<int>(h_v.size()), max); i++) {
    printf(format, h_v[i]);
  }
  std::cout << "\n";
}

template <typename T1, typename T2>
T1 div_round_up(const T1 a, const T2 b) {
  return static_cast<T1>(ceil(static_cast<double>(a) / b));
}

template <typename T>
thrust::device_ptr<T> dptr(T *d_ptr) {
  return thrust::device_pointer_cast(d_ptr);
}

template <typename T>
T *raw(thrust::device_vector<T> &v) {  //  NOLINT
  return raw_pointer_cast(v.data());
}

template <typename T>
const T *raw(const thrust::device_vector<T> &v) {  //  NOLINT
  return raw_pointer_cast(v.data());
}

template <typename T>
size_t size_bytes(const thrust::device_vector<T> &v) {
  return sizeof(T) * v.size();
}
/*
 * Kernel launcher
 */

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
  safe_cuda(cudaSetDevice(device_idx));
  const int GRID_SIZE = div_round_up(n, ITEMS_PER_THREAD * BLOCK_THREADS);
#if defined(__CUDACC__)
  launch_n_kernel<<<GRID_SIZE, BLOCK_THREADS>>>(static_cast<size_t>(0), n,
                                                lambda);
#endif
}

// if n_devices=-1, then use all visible devices
template <int ITEMS_PER_THREAD = 8, int BLOCK_THREADS = 256, typename L>
inline void multi_launch_n(size_t n, int n_devices, L lambda) {
  n_devices = n_devices < 0 ? n_visible_devices() : n_devices;
  CHECK_LE(n_devices, n_visible_devices()) << "Number of devices requested "
                                              "needs to be less than equal to "
                                              "number of visible devices.";
  const int GRID_SIZE = div_round_up(n, ITEMS_PER_THREAD * BLOCK_THREADS);
#if defined(__CUDACC__)
  n_devices = n_devices > n ? n : n_devices;
  for (int device_idx = 0; device_idx < n_devices; device_idx++) {
    safe_cuda(cudaSetDevice(device_idx));
    size_t begin = (n / n_devices) * device_idx;
    size_t end = std::min((n / n_devices) * (device_idx + 1), n);
    launch_n_kernel<<<GRID_SIZE, BLOCK_THREADS>>>(device_idx, begin, end,
                                                  lambda);
  }
#endif
}

/*
 * Random
 */

struct BernoulliRng {
  float p;
  int seed;

  __host__ __device__ BernoulliRng(float p, int seed) : p(p), seed(seed) {}

  __host__ __device__ bool operator()(const int i) const {
    thrust::default_random_engine rng(seed);
    thrust::uniform_real_distribution<float> dist;
    rng.discard(i);
    return dist(rng) <= p;
  }
};

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
void FindMergePartitions(int device_idx, coordinate_t *d_tile_coordinates, int num_tiles,
                         int tile_size, segments_t segments, offset_t num_rows,
                         offset_t num_elements) {
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
        segment_end_offsets[min(tile_start_coord.x + item, num_segments - 1)];
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

/**
 * \fn  template <typename func_t, typename segments_iter, typename offset_t>
 * void TransformLbs(int device_idx, dh::CubMemory *temp_memory, offset_t count,
 * segments_iter segments, offset_t num_segments, func_t f)
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
 * \tparam  segments_t    Type of the segments t.
 * \param           device_idx    Zero-based index of the device.
 * \param [in,out]  temp_memory   Temporary memory allocator.
 * \param           count         Number of elements.
 * \param           segments      Device pointer to segments.
 * \param           num_segments  Number of segments.
 * \param           f             Lambda to be executed on matrix elements.
 */

template <typename func_t, typename segments_iter, typename offset_t>
void TransformLbs(int device_idx, dh::CubMemory *temp_memory, offset_t count,
                  segments_iter segments, offset_t num_segments, func_t f) {
  typedef typename cub::CubVector<offset_t, 2>::Type coordinate_t;
  dh::safe_cuda(cudaSetDevice(device_idx));
  const int BLOCK_THREADS = 256;
  const int ITEMS_PER_THREAD = 1;
  const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
  int num_tiles = dh::div_round_up(count + num_segments, BLOCK_THREADS);

  temp_memory->LazyAllocate(sizeof(coordinate_t) * (num_tiles + 1));
  coordinate_t *tmp_tile_coordinates =
      reinterpret_cast<coordinate_t *>(temp_memory->d_temp_storage);

  FindMergePartitions(device_idx, tmp_tile_coordinates, num_tiles, BLOCK_THREADS, segments,
                      num_segments, count);

  LbsKernel<TILE_SIZE, ITEMS_PER_THREAD, BLOCK_THREADS, offset_t>
      <<<num_tiles, BLOCK_THREADS>>>(tmp_tile_coordinates, segments + 1, f,
                                     num_segments);
}

}  // namespace dh
