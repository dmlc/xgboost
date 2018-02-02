/*!
 * Copyright 2017 XGBoost contributors
 */
#ifndef XGBOOST_USE_CUDA

// dummy implementation of HostDeviceVector in case CUDA is not used

#include <xgboost/base.h>
#include "./host_device_vector.h"

namespace xgboost {

template <typename T>
struct HostDeviceVectorImpl {
  explicit HostDeviceVectorImpl(size_t size) : data_h_(size) {}
  std::vector<T> data_h_;
};

template <typename T>
HostDeviceVector<T>::HostDeviceVector(size_t size, int device) : impl_(nullptr) {
  impl_ = new HostDeviceVectorImpl<T>(size);
}

template <typename T>
HostDeviceVector<T>::~HostDeviceVector() {
  HostDeviceVectorImpl<T>* tmp = impl_;
  impl_ = nullptr;
  delete tmp;
}

template <typename T>
size_t HostDeviceVector<T>::size() const { return impl_->data_h_.size(); }

template <typename T>
int HostDeviceVector<T>::device() const { return -1; }

template <typename T>
T* HostDeviceVector<T>::ptr_d(int device) { return nullptr; }

template <typename T>
std::vector<T>& HostDeviceVector<T>::data_h() { return impl_->data_h_; }

template <typename T>
void HostDeviceVector<T>::resize(size_t new_size, int new_device) {
  impl_->data_h_.resize(new_size);
}

// explicit instantiations are required, as HostDeviceVector isn't header-only
template class HostDeviceVector<bst_float>;
template class HostDeviceVector<bst_gpair>;

}  // namespace xgboost

#endif
