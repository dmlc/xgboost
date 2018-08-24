/*!
 * Copyright 2017 XGBoost contributors
 */
#ifndef XGBOOST_USE_CUDA

// dummy implementation of HostDeviceVector in case CUDA is not used

#include <xgboost/base.h>
#include <xgboost/data.h>
#include <cstdint>
#include <utility>
#include "./host_device_vector.h"

namespace xgboost {

template <typename T>
struct HostDeviceVectorImpl {
  explicit HostDeviceVectorImpl(size_t size, T v) : data_h_(size, v), distribution_() {}
  HostDeviceVectorImpl(std::initializer_list<T> init) : data_h_(init), distribution_() {}
  explicit HostDeviceVectorImpl(std::vector<T>  init) : data_h_(std::move(init)), distribution_() {}
  std::vector<T> data_h_;
  GPUDistribution distribution_;
};

template <typename T>
HostDeviceVector<T>::HostDeviceVector(size_t size, T v, GPUDistribution distribution)
  : impl_(nullptr) {
  impl_ = new HostDeviceVectorImpl<T>(size, v);
}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(std::initializer_list<T> init, GPUDistribution distribution)
  : impl_(nullptr) {
  impl_ = new HostDeviceVectorImpl<T>(init);
}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(const std::vector<T>& init, GPUDistribution distribution)
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
HostDeviceVector<T>::HostDeviceVector(const HostDeviceVector<T>& other)
  : impl_(nullptr) {
  impl_ = new HostDeviceVectorImpl<T>(*other.impl_);
}

template <typename T>
HostDeviceVector<T>& HostDeviceVector<T>::operator=(const HostDeviceVector<T>& other) {
  if (this == &other) {
    return *this;
  }
  delete impl_;
  impl_ = new HostDeviceVectorImpl<T>(*other.impl_);
  return *this;
}

template <typename T>
size_t HostDeviceVector<T>::Size() const { return impl_->data_h_.size(); }

template <typename T>
GPUSet HostDeviceVector<T>::Devices() const { return GPUSet::Empty(); }

template <typename T>
const GPUDistribution& HostDeviceVector<T>::Distribution() const {
  return impl_->distribution_;
}

template <typename T>
T* HostDeviceVector<T>::DevicePointer(int device) { return nullptr; }

template <typename T>
const T* HostDeviceVector<T>::ConstDevicePointer(int device) const {
  return nullptr;
}

template <typename T>
common::Span<T> HostDeviceVector<T>::DeviceSpan(int device) {
  return common::Span<T>();
}

template <typename T>
common::Span<const T> HostDeviceVector<T>::ConstDeviceSpan(int device) const {
  return common::Span<const T>();
}

template <typename T>
std::vector<T>& HostDeviceVector<T>::HostVector() { return impl_->data_h_; }

template <typename T>
const std::vector<T>& HostDeviceVector<T>::ConstHostVector() const {
  return impl_->data_h_;
}

template <typename T>
void HostDeviceVector<T>::Resize(size_t new_size, T v) {
  impl_->data_h_.resize(new_size, v);
}

template <typename T>
size_t HostDeviceVector<T>::DeviceStart(int device) const { return 0; }

template <typename T>
size_t HostDeviceVector<T>::DeviceSize(int device) const { return 0; }

template <typename T>
void HostDeviceVector<T>::Fill(T v) {
  std::fill(HostVector().begin(), HostVector().end(), v);
}

template <typename T>
void HostDeviceVector<T>::Copy(const HostDeviceVector<T>& other) {
  CHECK_EQ(Size(), other.Size());
  std::copy(other.HostVector().begin(), other.HostVector().end(), HostVector().begin());
}

template <typename T>
void HostDeviceVector<T>::Copy(const std::vector<T>& other) {
  CHECK_EQ(Size(), other.size());
  std::copy(other.begin(), other.end(), HostVector().begin());
}

template <typename T>
void HostDeviceVector<T>::Copy(std::initializer_list<T> other) {
  CHECK_EQ(Size(), other.size());
  std::copy(other.begin(), other.end(), HostVector().begin());
}

template <typename T>
bool HostDeviceVector<T>::HostCanAccess(GPUAccess access) const {
  return true;
}

template <typename T>
bool HostDeviceVector<T>::DeviceCanAccess(int device, GPUAccess access) const {
  return false;
}

template <typename T>
void HostDeviceVector<T>::Reshard(const GPUDistribution& distribution) const { }

template <typename T>
void HostDeviceVector<T>::Reshard(GPUSet devices) const { }

// explicit instantiations are required, as HostDeviceVector isn't header-only
template class HostDeviceVector<bst_float>;
template class HostDeviceVector<GradientPair>;
template class HostDeviceVector<int>;
template class HostDeviceVector<Entry>;
template class HostDeviceVector<size_t>;

}  // namespace xgboost

#endif
