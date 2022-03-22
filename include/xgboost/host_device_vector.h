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
 * allocate/resize memory explicitly, and use the 'SetDevice' method
 * to specify the device.
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
 */

#ifndef XGBOOST_HOST_DEVICE_VECTOR_H_
#define XGBOOST_HOST_DEVICE_VECTOR_H_

#include <initializer_list>
#include <vector>
#include <type_traits>

#include "span.h"

namespace xgboost {

#ifdef __CUDACC__
// Sets a function to call instead of cudaSetDevice();
// only added for testing
void SetCudaSetDeviceHandler(void (*handler)(int));
#endif  // __CUDACC__

template <typename T> struct HostDeviceVectorImpl;

/*!
 * \brief Controls data access from the GPU.
 *
 * Since a `HostDeviceVector` can have data on both the host and device, access control needs to be
 * maintained to keep the data consistent.
 *
 * There are 3 scenarios supported:
 *   - Data is being manipulated on device. GPU has write access, host doesn't have access.
 *   - Data is read-only on both the host and device.
 *   - Data is being manipulated on the host. Host has write access, device doesn't have access.
 */
enum GPUAccess {
  kNone, kRead,
  // write implies read
  kWrite
};

template <typename T>
class HostDeviceVector {
  static_assert(std::is_standard_layout<T>::value, "HostDeviceVector admits only POD types");

 public:
  explicit HostDeviceVector(size_t size = 0, T v = T(), int device = -1);
  HostDeviceVector(std::initializer_list<T> init, int device = -1);
  explicit HostDeviceVector(const std::vector<T>& init, int device = -1);
  ~HostDeviceVector();

  HostDeviceVector(const HostDeviceVector<T>&) = delete;
  HostDeviceVector(HostDeviceVector<T>&&);

  HostDeviceVector<T>& operator=(const HostDeviceVector<T>&) = delete;
  HostDeviceVector<T>& operator=(HostDeviceVector<T>&&);

  bool Empty() const { return Size() == 0; }
  size_t Size() const;
  int DeviceIdx() const;
  common::Span<T> DeviceSpan();
  common::Span<const T> ConstDeviceSpan() const;
  common::Span<const T> DeviceSpan() const { return ConstDeviceSpan(); }
  T* DevicePointer();
  const T* ConstDevicePointer() const;
  const T* DevicePointer() const { return ConstDevicePointer(); }

  T* HostPointer() { return HostVector().data(); }
  common::Span<T> HostSpan() { return common::Span<T>{HostVector()}; }
  common::Span<T const> HostSpan() const { return common::Span<T const>{HostVector()}; }
  common::Span<T const> ConstHostSpan() const { return HostSpan(); }
  const T* ConstHostPointer() const { return ConstHostVector().data(); }
  const T* HostPointer() const { return ConstHostPointer(); }

  void Fill(T v);
  void Copy(const HostDeviceVector<T>& other);
  void Copy(const std::vector<T>& other);
  void Copy(std::initializer_list<T> other);

  void Extend(const HostDeviceVector<T>& other);

  std::vector<T>& HostVector();
  const std::vector<T>& ConstHostVector() const;
  const std::vector<T>& HostVector() const {return ConstHostVector(); }

  bool HostCanRead() const;
  bool HostCanWrite() const;
  bool DeviceCanRead() const;
  bool DeviceCanWrite() const;
  GPUAccess DeviceAccess() const;

  void SetDevice(int device) const;

  void Resize(size_t new_size, T v = T());

  using value_type = T;  // NOLINT

 private:
  HostDeviceVectorImpl<T>* impl_;
};

}  // namespace xgboost

#endif  // XGBOOST_HOST_DEVICE_VECTOR_H_
