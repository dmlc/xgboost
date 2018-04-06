/*!
 * Copyright 2017 XGBoost contributors
 */
#ifndef XGBOOST_USE_CUDA

// dummy implementation of HostDeviceVector in case CUDA is not used

#include <xgboost/base.h>

#include <utility>
#include "./host_device_vector.h"

namespace xgboost {

template <typename T>
struct HostDeviceVectorImpl {
  explicit HostDeviceVectorImpl(size_t size, T v) : data_h_(size, v) {}
  HostDeviceVectorImpl(std::initializer_list<T> init) : data_h_(init) {}
  explicit HostDeviceVectorImpl(std::vector<T>  init) : data_h_(std::move(init)) {}
  std::vector<T> data_h_;
};

template <typename T>
HostDeviceVector<T>::HostDeviceVector(size_t size, T v, int device) : impl_(nullptr) {
  impl_ = new HostDeviceVectorImpl<T>(size, v);
}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(std::initializer_list<T> init, int device)
  : impl_(nullptr) {
  impl_ = new HostDeviceVectorImpl<T>(init);
}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(const std::vector<T>& init, int device)
  : impl_(nullptr) {
  impl_ = new HostDeviceVectorImpl<T>(init);
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
void HostDeviceVector<T>::resize(size_t new_size, T v, int new_device) {
  impl_->data_h_.resize(new_size, v);
}

// explicit instantiations are required, as HostDeviceVector isn't header-only
template class HostDeviceVector<bst_float>;
template class HostDeviceVector<bst_gpair>;

}  // namespace xgboost

#endif
