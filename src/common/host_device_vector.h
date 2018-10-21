/*!
 * Copyright 2017 XGBoost contributors
 */

/**
 * @file host_device_vector.h
 * @brief A device-and-host vector abstraction layer.
 *
 * Why HostDeviceVector?<br/>
 * With CUDA, one has to explicitly manage memory through 'cudaMemcpy' calls.
 * This wrapper class hides this management from the users, thereby making it
 * easy to integrate GPU/CPU usage under a single interface.
 *
 * Initialization/Allocation:<br/>
 * One can choose to initialize the vector on CPU or GPU during constructor.
 * (use the 'devices' argument) Or, can choose to use the 'Resize' method to
 * allocate/resize memory explicitly, and use the 'Reshard' method
 * to specify the devices.
 *
 * Accessing underlying data:<br/>
 * Use 'HostVector' method to explicitly query for the underlying std::vector.
 * If you need the raw device pointer, use the 'DevicePointer' method. For perf
 * implications of these calls, see below.
 *
 * Accessing underling data and their perf implications:<br/>
 * There are 4 scenarios to be considered here:
 * HostVector and data on CPU --> no problems, std::vector returned immediately
 * HostVector but data on GPU --> this causes a cudaMemcpy to be issued internally.
 *                        subsequent calls to HostVector, will NOT incur this penalty.
 *                        (assuming 'DevicePointer' is not called in between)
 * DevicePointer but data on CPU  --> this causes a cudaMemcpy to be issued internally.
 *                        subsequent calls to DevicePointer, will NOT incur this penalty.
 *                        (assuming 'HostVector' is not called in between)
 * DevicePointer and data on GPU  --> no problems, the device ptr
 *                        will be returned immediately.
 *
 * What if xgboost is compiled without CUDA?<br/>
 * In that case, there's a special implementation which always falls-back to
 * working with std::vector. This logic can be found in host_device_vector.cc
 *
 * Why not consider CUDA unified memory?<br/>
 * We did consider. However, it poses complications if we need to support both
 * compiling with and without CUDA toolkit. It was easier to have
 * 'HostDeviceVector' with a special-case implementation in host_device_vector.cc
 *
 * @note: Size and Devices methods are thread-safe.
 * DevicePointer, DeviceStart, DeviceSize, tbegin and tend methods are thread-safe
 * if different threads call these methods with different values of the device argument.
 * All other methods are not thread safe.
 */

#ifndef XGBOOST_COMMON_HOST_DEVICE_VECTOR_H_
#define XGBOOST_COMMON_HOST_DEVICE_VECTOR_H_

#include <dmlc/logging.h>

#include <algorithm>
#include <cstdlib>
#include <initializer_list>
#include <vector>

#include "common.h"
#include "span.h"

// only include thrust-related files if host_device_vector.h
// is included from a .cu file
#ifdef __CUDACC__
#include <thrust/device_ptr.h>
#endif

namespace xgboost {

#ifdef __CUDACC__
// Sets a function to call instead of cudaSetDevice();
// only added for testing
void SetCudaSetDeviceHandler(void (*handler)(int));
#endif

template <typename T> struct HostDeviceVectorImpl;

// Distribution for the HostDeviceVector; it specifies such aspects as the devices it is
// distributed on, whether there are copies of elements from other GPUs as well as the granularity
// of splitting. It may also specify explicit boundaries for devices, in which case the size of the
// array cannot be changed.
class GPUDistribution {
  template<typename T> friend struct HostDeviceVectorImpl;

 public:
  explicit GPUDistribution(GPUSet devices = GPUSet::Empty())
    : devices_(devices), granularity_(1), overlap_(0) {}

 private:
  GPUDistribution(GPUSet devices, int granularity, int overlap,
                  std::vector<size_t> offsets)
    : devices_(devices), granularity_(granularity), overlap_(overlap),
    offsets_(std::move(offsets)) {}

 public:
  static GPUDistribution Block(GPUSet devices) { return GPUDistribution(devices); }

  static GPUDistribution Overlap(GPUSet devices, int overlap) {
    return GPUDistribution(devices, 1, overlap, std::vector<size_t>());
  }

  static GPUDistribution Granular(GPUSet devices, int granularity) {
    return GPUDistribution(devices, granularity, 0, std::vector<size_t>());
  }

  static GPUDistribution Explicit(GPUSet devices, std::vector<size_t> offsets) {
    return GPUDistribution(devices, 1, 0, offsets);
  }

  friend bool operator==(const GPUDistribution& a, const GPUDistribution& b) {
    bool const res = a.devices_ == b.devices_ &&
                     a.granularity_ == b.granularity_ &&
                     a.overlap_ == b.overlap_ &&
                     a.offsets_ == b.offsets_;
    return res;
  }

  friend bool operator!=(const GPUDistribution& a, const GPUDistribution& b) {
    return !(a == b);
  }

  GPUSet Devices() const { return devices_; }

  bool IsEmpty() const { return devices_.IsEmpty(); }

  size_t ShardStart(size_t size, int index) const {
    if (size == 0) { return 0; }
    if (offsets_.size() > 0) {
      // explicit offsets are provided
      CHECK_EQ(offsets_.back(), size);
      return offsets_.at(index);
    }
    // no explicit offsets
    size_t begin = std::min(index * Portion(size), size);
    begin = begin > size ? size : begin;
    return begin;
  }

  size_t ShardSize(size_t size, int index) const {
    if (size == 0) { return 0; }
    if (offsets_.size() > 0) {
      // explicit offsets are provided
      CHECK_EQ(offsets_.back(), size);
      return offsets_.at(index + 1)  - offsets_.at(index) +
        (index == devices_.Size() - 1 ? overlap_ : 0);
    }
    size_t portion = Portion(size);
    size_t begin = std::min(index * portion, size);
    size_t end = std::min((index + 1) * portion + overlap_ * granularity_, size);
    return end - begin;
  }

  size_t ShardProperSize(size_t size, int index) const {
    if (size == 0) { return 0; }
    return ShardSize(size, index) - (devices_.Size() - 1 > index ? overlap_ : 0);
  }

  bool IsFixedSize() const { return !offsets_.empty(); }

 private:
  static size_t DivRoundUp(size_t a, size_t b) { return (a + b - 1) / b; }
  static size_t RoundUp(size_t a, size_t b) { return DivRoundUp(a, b) * b; }

  size_t Portion(size_t size) const {
    return RoundUp
      (DivRoundUp
       (std::max(static_cast<int64_t>(size - overlap_ * granularity_),
                 static_cast<int64_t>(1)),
        devices_.Size()), granularity_);
  }

  GPUSet devices_;
  int granularity_;
  int overlap_;
  // explicit offsets for the GPU parts, if any
  std::vector<size_t> offsets_;
};

enum GPUAccess {
  kNone, kRead,
  // write implies read
  kWrite
};

inline GPUAccess operator-(GPUAccess a, GPUAccess b) {
  return static_cast<GPUAccess>(static_cast<int>(a) - static_cast<int>(b));
}

template <typename T>
class HostDeviceVector {
 public:
  explicit HostDeviceVector(size_t size = 0, T v = T(),
                            GPUDistribution distribution = GPUDistribution());
  HostDeviceVector(std::initializer_list<T> init,
                   GPUDistribution distribution = GPUDistribution());
  explicit HostDeviceVector(const std::vector<T>& init,
                            GPUDistribution distribution = GPUDistribution());
  ~HostDeviceVector();
  HostDeviceVector(const HostDeviceVector<T>&);
  HostDeviceVector<T>& operator=(const HostDeviceVector<T>&);
  size_t Size() const;
  GPUSet Devices() const;
  const GPUDistribution& Distribution() const;
  common::Span<T> DeviceSpan(int device);
  common::Span<const T> ConstDeviceSpan(int device) const;
  common::Span<const T> DeviceSpan(int device) const { return ConstDeviceSpan(device); }
  T* DevicePointer(int device);
  const T* ConstDevicePointer(int device) const;
  const T* DevicePointer(int device) const { return ConstDevicePointer(device); }

  T* HostPointer() { return HostVector().data(); }
  const T* ConstHostPointer() const { return ConstHostVector().data(); }
  const T* HostPointer() const { return ConstHostPointer(); }

  size_t DeviceStart(int device) const;
  size_t DeviceSize(int device) const;

  // only define functions returning device_ptr
  // if HostDeviceVector.h is included from a .cu file
#ifdef __CUDACC__
  thrust::device_ptr<T> tbegin(int device);  // NOLINT
  thrust::device_ptr<T> tend(int device);  // NOLINT
  thrust::device_ptr<const T> tcbegin(int device) const;  // NOLINT
  thrust::device_ptr<const T> tcend(int device) const;  // NOLINT
  thrust::device_ptr<const T> tbegin(int device) const {  // NOLINT
    return tcbegin(device);
  }
  thrust::device_ptr<const T> tend(int device) const { return tcend(device); }  // NOLINT

  void ScatterFrom(thrust::device_ptr<const T> begin, thrust::device_ptr<const T> end);
  void GatherTo(thrust::device_ptr<T> begin, thrust::device_ptr<T> end) const;
#endif

  void Fill(T v);
  void Copy(const HostDeviceVector<T>& other);
  void Copy(const std::vector<T>& other);
  void Copy(std::initializer_list<T> other);

  std::vector<T>& HostVector();
  const std::vector<T>& ConstHostVector() const;
  const std::vector<T>& HostVector() const {return ConstHostVector(); }

  bool HostCanAccess(GPUAccess access) const;
  bool DeviceCanAccess(int device, GPUAccess access) const;

  /*!
   * \brief Specify memory distribution.
   *
   *   If GPUSet::Empty() is used, all data will be drawn back to CPU.
   */
  void Reshard(const GPUDistribution& distribution) const;
  void Reshard(GPUSet devices) const;
  void Resize(size_t new_size, T v = T());

 private:
  HostDeviceVectorImpl<T>* impl_;
};

}  // namespace xgboost

#endif  // XGBOOST_COMMON_HOST_DEVICE_VECTOR_H_
