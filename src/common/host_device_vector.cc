/*!
 * Copyright 2017-2018 XGBoost contributors
 */
#include "./host_device_vector.h"

namespace xgboost {

size_t GPUDistribution::ShardStart(size_t size, int index) const {
  if (size == 0) { return 0; }
  if (offsets_.size() > 0) {
    // explicit offsets are provided
    CHECK_EQ(offsets_.back(), size) << *this;
    return offsets_.at(index);
  }
  // no explicit offsets
  size_t begin = std::min(index * Portion(size), size);
  begin = begin > size ? size : begin;
  return begin;
}

size_t GPUDistribution::ShardSize(size_t size, int index) const {
  if (size == 0) { return 0; }
  if (offsets_.size() > 0) {
    // explicit offsets are provided
    CHECK_EQ(offsets_.back(), size) << *this;
    return offsets_.at(index + 1)  - offsets_.at(index) +
        (index == devices_.Size() - 1 ? overlap_ : 0);
  }
  size_t portion = Portion(size);
  size_t begin = std::min(index * portion, size);
  size_t end = std::min((index + 1) * portion + overlap_ * granularity_, size);
  return end - begin;
}

size_t GPUDistribution::ShardProperSize(size_t size, size_t index) const {
  if (size == 0) { return 0; }
  return ShardSize(size, index) - (devices_.Size() - 1 > index ? overlap_ : 0);
}

size_t GPUDistribution::Portion(size_t size) const {
  size_t res = RoundUp(
      DivRoundUp(
          std::max(static_cast<int64_t>(size - overlap_ * granularity_),
                   static_cast<int64_t>(1)),
          devices_.Size()), granularity_);
  CHECK(size == 0 || size == 1 || !(devices_.Size() == size && size == res))
      << "res: " << res
      << *this;
  return res;
}

std::ostream& operator<<(std::ostream& os, GPUDistribution const& dist) {
  os << "\n"
     << "\tgpu_id: " << *dist.Devices().begin() << ", "
     << "n_gpus: " << dist.Devices().Size() << "\n"
     << "\tgranularity: " << dist.granularity_ << "\n"
     << "\toverlap:" << dist.overlap_ << "\n";
  os << "\toffsets: [";
  for (auto offset : dist.offsets_) {
    os << offset << ", ";
  }
  os << "]\n";
  return os;
}

}  // namespace xgboost

#ifndef XGBOOST_USE_CUDA

// dummy implementation of HostDeviceVector in case CUDA is not used

#include <xgboost/base.h>
#include <xgboost/data.h>
#include <cstdint>
#include <utility>

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
