/*!
 * Copyright 2017 XGBoost contributors
 */
#ifndef XGBOOST_COMMON_DHVEC_H_
#define XGBOOST_COMMON_DHVEC_H_

#include <cstdlib>
#include <vector>

// only include thrust-related files if dhvec.h is included from a .cu file
#ifdef __CUDACC__
#include <thrust/device_ptr.h>
#endif

namespace xgboost {

template <typename T> struct dhvec_impl;

template <typename T>
  class dhvec {
public:
  dhvec(size_t size = 0, int device = -1);
  ~dhvec();
  dhvec(const dhvec<T>&) = delete;
  dhvec(dhvec<T>&&) = delete;
  void operator=(const dhvec<T>&) = delete;
  void operator=(dhvec<T>&&) = delete;
  size_t size() const;
  int device() const;
  T* ptr_d(int device);

  // only define functions returning device_ptr
  // if dhvec.h is included from a .cu file
#ifdef __CUDACC__
  thrust::device_ptr<T> tbegin(int device);
  thrust::device_ptr<T> tend(int device);
#endif

  std::vector<T>& data_h();
  void resize(size_t new_size, int new_device);

  // helper functions in case a function needs to be templated
  // to work for both dhvec and std::vector
  static std::vector<T>& data_h(dhvec<T>& v) {
    return v.data_h();
  }

  static std::vector<T>& data_h(std::vector<T>& v) {
    return v;
  }  
  
private:
  dhvec_impl<T>* impl_;
};

}  // namespace xgboost

#endif
