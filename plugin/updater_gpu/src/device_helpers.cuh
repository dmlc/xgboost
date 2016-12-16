/*!
 * Copyright 2016 Rory mitchell
*/
#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
#include <algorithm>
#include <ctime>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

// Uncomment to enable
// #define DEVICE_TIMER
// #define TIMERS

namespace dh {

/*
 * Error handling  functions
 */

#define safe_cuda(ans) throw_on_cuda_error((ans), __FILE__, __LINE__)

cudaError_t throw_on_cuda_error(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::stringstream ss;
    ss << file << "(" << line << ")";
    std::string file_and_line;
    ss >> file_and_line;
    throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
  }

  return code;
}

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

/*
 *  Timers
 */

#define MAX_WARPS 32  // Maximum number of warps to time
#define MAX_SLOTS 10
#define TIMER_BLOCKID 0  // Block to time
struct DeviceTimerGlobal {
#ifdef DEVICE_TIMER

  clock_t total_clocks[MAX_SLOTS][MAX_WARPS];
  int64_t count[MAX_SLOTS][MAX_WARPS];

#endif

  // Clear device memory. Call at start of kernel.
  __device__ void Init() {
#ifdef DEVICE_TIMER
    if (blockIdx.x == TIMER_BLOCKID && threadIdx.x < MAX_WARPS) {
      for (int SLOT = 0; SLOT < MAX_SLOTS; SLOT++) {
        total_clocks[SLOT][threadIdx.x] = 0;
        count[SLOT][threadIdx.x] = 0;
      }
    }
#endif
  }

  void HostPrint() {
#ifdef DEVICE_TIMER
    DeviceTimerGlobal h_timer;
    safe_cuda(
        cudaMemcpyFromSymbol(&h_timer, (*this), sizeof(DeviceTimerGlobal)));

    for (int SLOT = 0; SLOT < MAX_SLOTS; SLOT++) {
      if (h_timer.count[SLOT][0] == 0) {
        continue;
      }

      clock_t sum_clocks = 0;
      int64_t sum_count = 0;

      for (int WARP = 0; WARP < MAX_WARPS; WARP++) {
        if (h_timer.count[SLOT][WARP] == 0) {
          continue;
        }

        sum_clocks += h_timer.total_clocks[SLOT][WARP];
        sum_count += h_timer.count[SLOT][WARP];
      }

      printf("Slot %d: %d clocks per call, called %d times.\n", SLOT,
             sum_clocks / sum_count, h_timer.count[SLOT][0]);
    }
#endif
  }
};

struct DeviceTimer {
#ifdef DEVICE_TIMER
  clock_t start;
  int slot;
  DeviceTimerGlobal &GTimer;
#endif

#ifdef DEVICE_TIMER
  __device__ DeviceTimer(DeviceTimerGlobal &GTimer, int slot) // NOLINT
      : GTimer(GTimer), start(clock()), slot(slot) {}
#else
  __device__ DeviceTimer(DeviceTimerGlobal &GTimer, int slot) {} // NOLINT
#endif

  __device__ void End() {
#ifdef DEVICE_TIMER
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (blockIdx.x == TIMER_BLOCKID && lane_id == 0) {
      GTimer.count[slot][warp_id] += 1;
      GTimer.total_clocks[slot][warp_id] += clock() - start;
    }
#endif
  }
};

struct Timer {
  volatile double start;
  Timer() { reset(); }

  double seconds_now() {
#ifdef _WIN32
    static LARGE_INTEGER s_frequency;
    QueryPerformanceFrequency(&s_frequency);
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return static_cast<double>(now.QuadPart) / s_frequency.QuadPart;
#endif
  }

  void reset() {
#ifdef _WIN32
    _ReadWriteBarrier();
    start = seconds_now();
#endif
  }
  double elapsed() {
#ifdef _WIN32
    _ReadWriteBarrier();
    return seconds_now() - start;
#endif
  }
  void printElapsed(char *label) {
#ifdef TIMERS
    safe_cuda(cudaDeviceSynchronize());
    printf("%s:\t %1.4fs\n", label, elapsed());
#endif
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

template <typename T> __device__ range grid_stride_range(T begin, T end) {
  begin += blockDim.x * blockIdx.x + threadIdx.x;
  range r(begin, end);
  r.step(gridDim.x * blockDim.x);
  return r;
}

template <typename T> __device__ range block_stride_range(T begin, T end) {
  begin += threadIdx.x;
  range r(begin, end);
  r.step(blockDim.x);
  return r;
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
void print(char *label, const thrust::device_vector<T> &v,
           const char *format = "%d ", int max = 10) {
  thrust::host_vector<T> h_v = v;

  std::cout << label << ":\n";
  for (int i = 0; i < std::min(static_cast<int>(h_v.size()), max); i++) {
    printf(format, h_v[i]);
  }
  std::cout << "\n";
}

template <typename T1, typename T2> T1 div_round_up(const T1 a, const T2 b) {
  return static_cast<T1>(ceil(static_cast<double>(a) / b));
}

template <typename T> thrust::device_ptr<T> dptr(T *d_ptr) {
  return thrust::device_pointer_cast(d_ptr);
}

template <typename T> T *raw(thrust::device_vector<T> &v) { //  NOLINT
  return raw_pointer_cast(v.data());
}

template <typename T> size_t size_bytes(const thrust::device_vector<T> &v) {
  return sizeof(T) * v.size();
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

class bulk_allocator;

template <typename T> class dvec {
  friend bulk_allocator;

 private:
  T *_ptr;
  size_t _size;

  void external_allocate(void *ptr, size_t size) {
    if (!empty()) {
      throw std::runtime_error("Tried to allocate dvec but already allocated");
    }

    _ptr = static_cast<T *>(ptr);
    _size = size;
  }

 public:
  dvec() : _ptr(NULL), _size(0) {}
  size_t size() { return _size; }
  bool empty() { return _ptr == NULL || _size == 0; }
  T *data() { return _ptr; }

  std::vector<T> as_vector() {
    std::vector<T> h_vector(size());
    safe_cuda(cudaMemcpy(h_vector.data(), _ptr, size() * sizeof(T),
                         cudaMemcpyDeviceToHost));
    return h_vector;
  }

  void fill(T value) {
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

  template <typename T2> dvec &operator=(const std::vector<T2> &other) {
    if (other.size() != size()) {
      throw std::runtime_error(
          "Cannot copy assign vector to dvec, sizes are different");
    }

    thrust::copy(other.begin(), other.end(), this->tbegin());

    return *this;
  }

  dvec &operator=(dvec<T> &other) {
    if (other.size() != size()) {
      throw std::runtime_error(
          "Cannot copy assign dvec to dvec, sizes are different");
    }

    thrust::copy(other.tbegin(), other.tend(), this->tbegin());

    return *this;
  }
};

class bulk_allocator {
  char *d_ptr;
  size_t _size;

  const size_t align = 256;

  template <typename SizeT> size_t align_round_up(SizeT n) {
    if (n % align == 0) {
      return n;
    } else {
      return n + align - (n % align);
    }
  }

  template <typename T, typename SizeT>
  size_t get_size_bytes(dvec<T> *first_vec, SizeT first_size) {
    return align_round_up(first_size * sizeof(T));
  }

  template <typename T, typename SizeT, typename... Args>
  size_t get_size_bytes(dvec<T> *first_vec, SizeT first_size, Args... args) {
    return align_round_up(first_size * sizeof(T)) + get_size_bytes(args...);
  }

  template <typename T, typename SizeT>
  void allocate_dvec(char *ptr, dvec<T> *first_vec, SizeT first_size) {
    first_vec->external_allocate(static_cast<void *>(ptr), first_size);
  }

  template <typename T, typename SizeT, typename... Args>
  void allocate_dvec(char *ptr, dvec<T> *first_vec, SizeT first_size,
                     Args... args) {
    first_vec->external_allocate(static_cast<void*>(ptr), first_size);
    ptr += align_round_up(first_size * sizeof(T));
    allocate_dvec(ptr, args...);
  }

 public:
  bulk_allocator() : _size(0), d_ptr(NULL) {}

  ~bulk_allocator() {
    if (!d_ptr == NULL) {
      safe_cuda(cudaFree(d_ptr));
    }
  }

  size_t size() { return _size; }

  template <typename... Args> void allocate(Args... args) {
    if (d_ptr != NULL) {
      throw std::runtime_error("Bulk allocator already allocated");
    }

    _size = get_size_bytes(args...);

    safe_cuda(cudaMalloc(&d_ptr, _size));

    allocate_dvec(d_ptr, args...);
  }
};

// Keep track of cub library device allocation
struct CubMemory {
  void *d_temp_storage;
  size_t temp_storage_bytes;

  CubMemory() : d_temp_storage(NULL), temp_storage_bytes(0) {}

  ~CubMemory() {
    if (d_temp_storage != NULL) {
      safe_cuda(cudaFree(d_temp_storage));
    }
  }

  void Allocate() {
    safe_cuda(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  }

  bool IsAllocated() { return d_temp_storage != NULL; }
};

inline size_t available_memory() {
  size_t device_free = 0;
  size_t device_total = 0;
  dh::safe_cuda(cudaMemGetInfo(&device_free, &device_total));
  return device_free;
}

inline std::string device_name() {
  cudaDeviceProp prop;
  dh::safe_cuda(cudaGetDeviceProperties(&prop, 0));
  return std::string(prop.name);
}

/*
 * Kernel launcher
 */

template <typename L> __global__ void launch_n_kernel(size_t n, L lambda) {
  for (auto i : grid_stride_range(static_cast<size_t>(0), n)) {
    lambda(i);
  }
}

template <typename L, int ITEMS_PER_THREAD = 8, int BLOCK_THREADS = 256>
inline void launch_n(size_t n, L lambda) {
  const int GRID_SIZE = div_round_up(n, ITEMS_PER_THREAD * BLOCK_THREADS);

  launch_n_kernel<<<GRID_SIZE, BLOCK_THREADS>>>(n, lambda);
}
}  // namespace dh
