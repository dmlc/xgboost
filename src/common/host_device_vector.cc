/*!
 * Copyright 2017 XGBoost contributors
 */
#ifndef XGBOOST_USE_CUDA

// dummy implementation of HostDeviceVector in case CUDA is not used

#include <xgboost/base.h>
#include <xgboost/data.h>
#include <cstdint>
#include <memory>
#include <utility>
#include "xgboost/tree_model.h"
#include "xgboost/host_device_vector.h"

namespace xgboost {

template <typename T>
struct HostDeviceVectorImpl {
  explicit HostDeviceVectorImpl(size_t size, T v) : data_h_(size, v) {}
  HostDeviceVectorImpl(std::initializer_list<T> init) : data_h_(init) {}
  explicit HostDeviceVectorImpl(std::vector<T>  init) : data_h_(std::move(init)) {}
  HostDeviceVectorImpl(HostDeviceVectorImpl&& that) : data_h_(std::move(that.data_h_)) {}

  void Swap(HostDeviceVectorImpl &other) {
     data_h_.swap(other.data_h_);
  }

  std::vector<T>& Vec() { return data_h_; }

 private:
  std::vector<T> data_h_;
};

template <typename T>
HostDeviceVector<T>::HostDeviceVector(size_t size, T v, int)
  : impl_(nullptr) {
  impl_ = new HostDeviceVectorImpl<T>(size, v);
}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(std::initializer_list<T> init, int)
  : impl_(nullptr) {
  impl_ = new HostDeviceVectorImpl<T>(init);
}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(const std::vector<T>& init, int)
  : impl_(nullptr) {
  impl_ = new HostDeviceVectorImpl<T>(init);
}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(HostDeviceVector<T>&& that) {
  impl_ = new HostDeviceVectorImpl<T>(std::move(*that.impl_));
}

template <typename T>
HostDeviceVector<T>& HostDeviceVector<T>::operator=(HostDeviceVector<T>&& that) {
  if (this == &that) { return *this; }

  std::unique_ptr<HostDeviceVectorImpl<T>> new_impl(
      new HostDeviceVectorImpl<T>(std::move(*that.impl_)));
  delete impl_;
  impl_ = new_impl.release();
  return *this;
}

template <typename T>
HostDeviceVector<T>::~HostDeviceVector() {
  delete impl_;
  impl_ = nullptr;
}

template <typename T>
GPUAccess HostDeviceVector<T>::DeviceAccess() const {
  return kNone;
}

template <typename T>
size_t HostDeviceVector<T>::Size() const { return impl_->Vec().size(); }

template <typename T>
int HostDeviceVector<T>::DeviceIdx() const { return -1; }

template <typename T>
T* HostDeviceVector<T>::DevicePointer() { return nullptr; }

template <typename T>
const T* HostDeviceVector<T>::ConstDevicePointer() const {
  return nullptr;
}

template <typename T>
common::Span<T> HostDeviceVector<T>::DeviceSpan() {
  return common::Span<T>();
}

template <typename T>
common::Span<const T> HostDeviceVector<T>::ConstDeviceSpan() const {
  return common::Span<const T>();
}

template <typename T>
std::vector<T>& HostDeviceVector<T>::HostVector() { return impl_->Vec(); }

template <typename T>
const std::vector<T>& HostDeviceVector<T>::ConstHostVector() const {
  return impl_->Vec();
}

template <typename T>
void HostDeviceVector<T>::Resize(size_t new_size, T v) {
  impl_->Vec().resize(new_size, v);
}

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
void HostDeviceVector<T>::Extend(HostDeviceVector const& other) {
  auto ori_size = this->Size();
  this->HostVector().resize(ori_size + other.Size());
  std::copy(other.ConstHostVector().cbegin(), other.ConstHostVector().cend(),
            this->HostVector().begin() + ori_size);
}

template <typename T>
bool HostDeviceVector<T>::HostCanRead() const {
  return true;
}

template <typename T>
bool HostDeviceVector<T>::HostCanWrite() const {
  return true;
}

template <typename T>
bool HostDeviceVector<T>::DeviceCanRead() const {
  return false;
}

template <typename T>
bool HostDeviceVector<T>::DeviceCanWrite() const {
  return false;
}

template <typename T>
void HostDeviceVector<T>::SetDevice(int) const {}

// explicit instantiations are required, as HostDeviceVector isn't header-only
template class HostDeviceVector<bst_float>;
template class HostDeviceVector<double>;
template class HostDeviceVector<GradientPair>;
template class HostDeviceVector<int32_t>;   // bst_node_t
template class HostDeviceVector<uint8_t>;
template class HostDeviceVector<FeatureType>;
template class HostDeviceVector<Entry>;
template class HostDeviceVector<uint64_t>;  // bst_row_t
template class HostDeviceVector<uint32_t>;  // bst_feature_t
template class HostDeviceVector<RegTree::Segment>;

#if defined(__APPLE__) || defined(__EMSCRIPTEN__)
/*
 * On OSX:
 *
 * typedef unsigned int         uint32_t;
 * typedef unsigned long long   uint64_t;
 * typedef unsigned long       __darwin_size_t;
 *
 * On Emscripten:
 * typedef unsigned long        size_t;
 */
template class HostDeviceVector<std::size_t>;
#endif  // defined(__APPLE__)

}  // namespace xgboost

#endif  // XGBOOST_USE_CUDA
