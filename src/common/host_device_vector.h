/*!
 * Copyright 2017 XGBoost contributors
 */
#ifndef XGBOOST_COMMON_HOST_DEVICE_VECTOR_H_
#define XGBOOST_COMMON_HOST_DEVICE_VECTOR_H_

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
 * (use the 'device' argument) Or, can choose to use the 'resize' method to
 * allocate/resize memory explicitly.
 *
 * Accessing underling data:<br/>
 * Use 'data_h' method to explicitly query for the underlying std::vector.
 * If you need the raw device pointer, use the 'ptr_d' method. For perf
 * implications of these calls, see below.
 *
 * Accessing underling data and their perf implications:<br/>
 * There are 4 scenarios to be considered here:
 * data_h and data on CPU --> no problems, std::vector returned immediately
 * data_h but data on GPU --> this causes a cudaMemcpy to be issued internally.
 *                        subsequent calls to data_h, will NOT incur this penalty.
 *                        (assuming 'ptr_d' is not called in between)
 * ptr_d but data on CPU  --> this causes a cudaMemcpy to be issued internally.
 *                        subsequent calls to ptr_d, will NOT incur this penalty.
 *                        (assuming 'data_h' is not called in between)
 * ptr_d and data on GPU  --> no problems, the device ptr will be returned immediately
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
 * @note: This is not thread-safe!
 */
template <typename T>
class HostDeviceVector {
 public:
  explicit HostDeviceVector(size_t size = 0, int device = -1);
  HostDeviceVector(std::initializer_list<T> init, int device = -1);
  ~HostDeviceVector();
  HostDeviceVector(const HostDeviceVector<T>&) = delete;
  HostDeviceVector(HostDeviceVector<T>&&) = delete;
  void operator=(const HostDeviceVector<T>&) = delete;
  void operator=(HostDeviceVector<T>&&) = delete;
  size_t size() const;
  int device() const;
  T* ptr_d(int device);
  T* ptr_h() { return data_h().data(); }

  // only define functions returning device_ptr
  // if HostDeviceVector.h is included from a .cu file
#ifdef __CUDACC__
  thrust::device_ptr<T> tbegin(int device);
  thrust::device_ptr<T> tend(int device);
#endif

  std::vector<T>& data_h();

  // passing in new_device == -1 keeps the device as is
  void resize(size_t new_size, int new_device);

  // helper functions in case a function needs to be templated
  // to work for both HostDeviceVector and std::vector
  static std::vector<T>& data_h(HostDeviceVector<T>* v) {
    return v->data_h();
  }

  static std::vector<T>& data_h(std::vector<T>* v) {
    return *v;
  }

 private:
  HostDeviceVectorImpl<T>* impl_;
};

}  // namespace xgboost

#endif  // XGBOOST_COMMON_HOST_DEVICE_VECTOR_H_
