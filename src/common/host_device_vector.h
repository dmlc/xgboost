/*!
 * Copyright 2017 XGBoost contributors
 */
#ifndef XGBOOST_COMMON_HOST_DEVICE_VECTOR_H_
#define XGBOOST_COMMON_HOST_DEVICE_VECTOR_H_

#include <dmlc/logging.h>

#include <algorithm>
#include <cstdlib>
#include <initializer_list>
#include <vector>

// only include thrust-related files if host_device_vector.h
// is included from a .cu file
#ifdef __CUDACC__
#include <thrust/device_ptr.h>
#endif

namespace xgboost {

template <typename T> struct HostDeviceVectorImpl;

// set of devices across which HostDeviceVector can be distributed;
// currently implemented as a range, but can be changed later to something else,
// e.g. a bitset
class GPUSet {
 public:
  explicit GPUSet(int start = 0, int ndevices = 0)
    : start_(start), ndevices_(ndevices) {}
  static GPUSet Empty() { return GPUSet(); }
  static GPUSet Range(int start, int ndevices) { return GPUSet(start, ndevices); }
  int Size() const { return ndevices_; }
  int operator[](int index) const {
    CHECK(index >= 0 && index < ndevices_);
    return start_ + index;
  }
  bool IsEmpty() const { return ndevices_ <= 0; }
  int Index(int device) const {
    CHECK(device >= start_ && device < start_ + ndevices_);
    return device - start_;
  }
  bool Contains(int device) const {
    return start_ <= device && device < start_ + ndevices_;
  }
  friend bool operator==(GPUSet a, GPUSet b) {
    return a.start_ == b.start_ && a.ndevices_ == b.ndevices_;
  }
  friend bool operator!=(GPUSet a, GPUSet b) {
    return a.start_ != b.start_ || a.ndevices_ != b.ndevices_;
  }

 private:
  int start_, ndevices_;
};


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
template <typename T>
class HostDeviceVector {
 public:
  explicit HostDeviceVector(size_t size = 0, T v = T(),
                            GPUSet devices = GPUSet::Empty());
  HostDeviceVector(std::initializer_list<T> init, GPUSet devices = GPUSet::Empty());
  explicit HostDeviceVector(const std::vector<T>& init,
                            GPUSet devices = GPUSet::Empty());
  ~HostDeviceVector();
  HostDeviceVector(const HostDeviceVector<T>&) = delete;
  HostDeviceVector(HostDeviceVector<T>&&) = delete;
  void operator=(const HostDeviceVector<T>&) = delete;
  void operator=(HostDeviceVector<T>&&) = delete;
  size_t Size() const;
  GPUSet Devices() const;
  T* DevicePointer(int device);

  T* HostPointer() { return HostVector().data(); }
  size_t DeviceStart(int device);
  size_t DeviceSize(int device);

  // only define functions returning device_ptr
  // if HostDeviceVector.h is included from a .cu file
#ifdef __CUDACC__
  thrust::device_ptr<T> tbegin(int device);  // NOLINT
  thrust::device_ptr<T> tend(int device);  // NOLINT
  void ScatterFrom(thrust::device_ptr<T> begin, thrust::device_ptr<T> end);
  void GatherTo(thrust::device_ptr<T> begin, thrust::device_ptr<T> end);
#endif

  void Fill(T v);
  void Copy(HostDeviceVector<T>* other);
  void Copy(const std::vector<T>& other);
  void Copy(std::initializer_list<T> other);

  std::vector<T>& HostVector();
  void Reshard(GPUSet devices);
  void Resize(size_t new_size, T v = T());

 private:
  HostDeviceVectorImpl<T>* impl_;
};

}  // namespace xgboost

#endif  // XGBOOST_COMMON_HOST_DEVICE_VECTOR_H_
