/*!
 * Copyright 2017-2019 XGBoost contributors
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
 * allocate/resize memory explicitly, and use the 'Shard' method
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
#include <utility>
#include <vector>

#include "common.h"
#include "span.h"

// only include thrust-related files if host_device_vector.h
// is included from a .cu file
#ifdef __CUDACC__
#include <thrust/device_ptr.h>
#endif  // __CUDACC__

namespace xgboost {

#ifdef __CUDACC__
// Sets a function to call instead of cudaSetDevice();
// only added for testing
void SetCudaSetDeviceHandler(void (*handler)(int));
#endif  // __CUDACC__

template <typename T> struct HostDeviceVectorImpl;

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
  explicit HostDeviceVector(size_t size = 0, T v = T(), int device = -1);
  HostDeviceVector(std::initializer_list<T> init, int device = -1);
  explicit HostDeviceVector(const std::vector<T>& init, int device = -1);
  ~HostDeviceVector();
  HostDeviceVector(const HostDeviceVector<T>&);
  HostDeviceVector<T>& operator=(const HostDeviceVector<T>&);
  size_t Size() const;
  int DeviceIdx() const;
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
#endif  // __CUDACC__

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
   */
  void Shard(int device) const;

  /*!
   * \brief Change memory distribution.
   */
  void Reshard(int device);

  void Resize(size_t new_size, T v = T());

 private:
  HostDeviceVectorImpl<T>* impl_;
};

}  // namespace xgboost

#endif  // XGBOOST_COMMON_HOST_DEVICE_VECTOR_H_
