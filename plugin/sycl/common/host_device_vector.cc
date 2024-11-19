/**
 * Copyright 2017-2024 by XGBoost contributors
 */

#ifdef XGBOOST_USE_SYCL

// implementation of HostDeviceVector with sycl support

#include <memory>
#include <utility>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-W#pragma-messages"
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#include "xgboost/host_device_vector.h"
#pragma GCC diagnostic pop

#include "../device_manager.h"
#include "../data.h"

namespace xgboost {
template <typename T>
class HostDeviceVectorImpl {
  using DeviceStorage = sycl::USMVector<T, sycl::MemoryType::on_device>;

 public:
  explicit HostDeviceVectorImpl(size_t size, T v, DeviceOrd device) : device_(device) {
    if (device.IsSycl()) {
      device_access_ = GPUAccess::kWrite;
      SetDevice();
      data_d_->Resize(qu_, size, v);
    } else {
      data_h_.resize(size, v);
    }
  }

  template <class Initializer>
  HostDeviceVectorImpl(const Initializer& init, DeviceOrd device) : device_(device) {
    if (device.IsSycl()) {
      device_access_ = GPUAccess::kWrite;

      ResizeDevice(init.size());
      Copy(init);
    } else {
      data_h_ = init;
    }
  }

  HostDeviceVectorImpl(HostDeviceVectorImpl<T>&& that) : device_{that.device_},
                                                         data_h_{std::move(that.data_h_)},
                                                         data_d_{std::move(that.data_d_)},
                                                         device_access_{that.device_access_} {}

  std::vector<T>& HostVector() {
    SyncHost(GPUAccess::kNone);
    return data_h_;
  }

  const std::vector<T>& ConstHostVector() {
    SyncHost(GPUAccess::kRead);
    return data_h_;
  }

  void SetDevice(DeviceOrd device) {
    if (device_ == device) { return; }
    if (device_.IsSycl()) {
      SyncHost(GPUAccess::kNone);
    }

    if (device_.IsSycl() && device.IsSycl()) {
      CHECK_EQ(device_, device)
          << "New device is different from previous one.";
    }
    device_ = device;
    if (device_.IsSycl()) {
      ResizeDevice(data_h_.size());
    }
  }

  template <typename... U>
  void Resize(size_t new_size, U&&... args) {
    if (new_size == Size()) {
      return;
    }
    if ((Size() == 0 && device_.IsSycl()) || (DeviceCanWrite() && device_.IsSycl())) {
      // fast on-device resize
      device_access_ = GPUAccess::kWrite;
      SetDevice();
      auto old_size = data_d_->Size();
      data_d_->Resize(qu_, new_size, std::forward<U>(args)...);
    } else {
      // resize on host
      SyncHost(GPUAccess::kNone);
      auto old_size = data_h_.size();
      data_h_.resize(new_size, std::forward<U>(args)...);
    }
  }

  void SyncHost(GPUAccess access) {
    if (HostCanAccess(access)) { return; }
    if (HostCanRead()) {
      // data is present, just need to deny access to the device
      device_access_ = access;
      return;
    }
    device_access_ = access;
    if (data_h_.size() != data_d_->Size()) { data_h_.resize(data_d_->Size()); }
    SetDevice();
    qu_->memcpy(data_h_.data(), data_d_->Data(), data_d_->Size() * sizeof(T)).wait();
  }

  void SyncDevice(GPUAccess access) {
    if (DeviceCanAccess(access)) { return; }
    if (DeviceCanRead()) {
      device_access_ = access;
      return;
    }
    // data is on the host
    ResizeDevice(data_h_.size());
    SetDevice();
    qu_->memcpy(data_d_->Data(), data_h_.data(), data_d_->Size() * sizeof(T)).wait();
    device_access_ = access;
  }

  bool HostCanAccess(GPUAccess access) const { return device_access_ <= access; }
  bool HostCanRead() const { return HostCanAccess(GPUAccess::kRead); }
  bool HostCanWrite() const { return HostCanAccess(GPUAccess::kNone); }
  bool DeviceCanAccess(GPUAccess access) const { return device_access_ >= access; }
  bool DeviceCanRead() const { return DeviceCanAccess(GPUAccess::kRead); }
  bool DeviceCanWrite() const { return DeviceCanAccess(GPUAccess::kWrite); }
  GPUAccess Access() const { return device_access_; }

  size_t Size() const {
    return HostCanRead() ? data_h_.size() : data_d_ ? data_d_->Size() : 0;
  }

  DeviceOrd Device() const { return device_; }

  T* DevicePointer() {
    SyncDevice(GPUAccess::kWrite);
    return data_d_->Data();
  }

  const T* ConstDevicePointer() {
    SyncDevice(GPUAccess::kRead);
    return data_d_->DataConst();
  }

  common::Span<T> DeviceSpan() {
    SyncDevice(GPUAccess::kWrite);
    return {this->DevicePointer(), Size()};
  }

  common::Span<const T> ConstDeviceSpan() {
    SyncDevice(GPUAccess::kRead);
    return {this->ConstDevicePointer(), Size()};
  }

  void Fill(T v) {
    if (HostCanWrite()) {
      std::fill(data_h_.begin(), data_h_.end(), v);
    } else {
      device_access_ = GPUAccess::kWrite;
      SetDevice();
      qu_->fill(data_d_->Data(), v, data_d_->Size()).wait();
    }
  }

  void Copy(HostDeviceVectorImpl<T>* other) {
    CHECK_EQ(Size(), other->Size());
    SetDevice(other->device_);
    // Data is on host.
    if (HostCanWrite() && other->HostCanWrite()) {
      std::copy(other->data_h_.begin(), other->data_h_.end(), data_h_.begin());
      return;
    }
    SetDevice();
    CopyToDevice(other);
  }

  void Copy(const std::vector<T>& other) {
    CHECK_EQ(Size(), other.size());
    if (HostCanWrite()) {
      std::copy(other.begin(), other.end(), data_h_.begin());
    } else {
      CopyToDevice(other.data());
    }
  }

  void Copy(std::initializer_list<T> other) {
    CHECK_EQ(Size(), other.size());
    if (HostCanWrite()) {
      std::copy(other.begin(), other.end(), data_h_.begin());
    } else {
      CopyToDevice(other.begin());
    }
  }

  void Extend(HostDeviceVectorImpl* other) {
    auto ori_size = this->Size();
    this->Resize(ori_size + other->Size(), T{});
    if (HostCanWrite() && other->HostCanRead()) {
      auto& h_vec = this->HostVector();
      auto& other_vec = other->HostVector();
      CHECK_EQ(h_vec.size(), ori_size + other->Size());
      std::copy(other_vec.cbegin(), other_vec.cend(), h_vec.begin() + ori_size);
    } else {
      auto ptr = other->ConstDevicePointer();
      SetDevice();
      CHECK_EQ(this->Device(), other->Device());
      qu_->memcpy(this->DevicePointer() + ori_size, ptr, other->Size() * sizeof(T)).wait();
    }
  }

 private:
  void ResizeDevice(size_t new_size) {
    if (data_d_ && new_size == data_d_->Size()) { return; }
    SetDevice();
    data_d_->Resize(qu_, new_size);
  }

  void SetDevice() {
    if (!qu_) {
      qu_ = device_manager_.GetQueue(device_);
    }
    if (!data_d_) {
      data_d_.reset(new DeviceStorage());
    }
  }

  void CopyToDevice(HostDeviceVectorImpl* other) {
    if (other->HostCanWrite()) {
      CopyToDevice(other->data_h_.data());
    } else {
      ResizeDevice(Size());
      device_access_ = GPUAccess::kWrite;
      SetDevice();
      qu_->memcpy(data_d_->Data(), other->data_d_->Data(), data_d_->Size() * sizeof(T)).wait();
    }
  }

  void CopyToDevice(const T* begin) {
    data_d_->ResizeNoCopy(qu_, Size());
    qu_->memcpy(data_d_->Data(), begin, data_d_->Size() * sizeof(T)).wait();
    device_access_ = GPUAccess::kWrite;
  }

  sycl::DeviceManager device_manager_;
  ::sycl::queue* qu_ = nullptr;
  DeviceOrd device_{DeviceOrd::CPU()};
  std::vector<T> data_h_{};
  std::unique_ptr<DeviceStorage> data_d_{};
  GPUAccess device_access_{GPUAccess::kNone};
};

template <typename T>
HostDeviceVector<T>::HostDeviceVector(size_t size, T v, DeviceOrd device)
  : impl_(nullptr) {
  impl_ = new HostDeviceVectorImpl<T>(size, v, device);
}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(std::initializer_list<T> init, DeviceOrd device)
  : impl_(nullptr) {
  impl_ = new HostDeviceVectorImpl<T>(init, device);
}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(const std::vector<T>& init, DeviceOrd device)
  : impl_(nullptr) {
  impl_ = new HostDeviceVectorImpl<T>(init, device);
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
size_t HostDeviceVector<T>::Size() const { return impl_->Size(); }

template <typename T>
DeviceOrd HostDeviceVector<T>::Device() const {
  return impl_->Device();
}

template <typename T>
T* HostDeviceVector<T>::DevicePointer() {
  return impl_->DevicePointer();
}

template <typename T>
const T* HostDeviceVector<T>::ConstDevicePointer() const {
  return impl_->ConstDevicePointer();
}

template <typename T>
common::Span<T> HostDeviceVector<T>::DeviceSpan() {
  return impl_->DeviceSpan();
}

template <typename T>
common::Span<const T> HostDeviceVector<T>::ConstDeviceSpan() const {
  return impl_->ConstDeviceSpan();
}

template <typename T>
std::vector<T>& HostDeviceVector<T>::HostVector() { return impl_->HostVector(); }

template <typename T>
const std::vector<T>& HostDeviceVector<T>::ConstHostVector() const {
  return impl_->ConstHostVector();
}

template <typename T>
void HostDeviceVector<T>::Resize(size_t new_size, T v) {
  impl_->Resize(new_size, v);
}

template <typename T>
void HostDeviceVector<T>::Resize(size_t new_size) {
  impl_->Resize(new_size);
}

template <typename T>
void HostDeviceVector<T>::Fill(T v) {
  impl_->Fill(v);
}

template <typename T>
void HostDeviceVector<T>::Copy(const HostDeviceVector<T>& other) {
  impl_->Copy(other.impl_);
}

template <typename T>
void HostDeviceVector<T>::Copy(const std::vector<T>& other) {
  impl_->Copy(other);
}

template <typename T>
void HostDeviceVector<T>::Copy(std::initializer_list<T> other) {
  impl_->Copy(other);
}

template <typename T>
void HostDeviceVector<T>::Extend(HostDeviceVector const& other) {
  impl_->Extend(other.impl_);
}

template <typename T>
bool HostDeviceVector<T>::HostCanRead() const {
  return impl_->HostCanRead();
}

template <typename T>
bool HostDeviceVector<T>::HostCanWrite() const {
  return impl_->HostCanWrite();
}

template <typename T>
bool HostDeviceVector<T>::DeviceCanRead() const {
  return impl_->DeviceCanRead();
}

template <typename T>
bool HostDeviceVector<T>::DeviceCanWrite() const {
  return impl_->DeviceCanWrite();
}

template <typename T>
GPUAccess HostDeviceVector<T>::DeviceAccess() const {
  return impl_->Access();
}

template <typename T>
void HostDeviceVector<T>::SetDevice(DeviceOrd device) const {
  impl_->SetDevice(device);
}

// explicit instantiations are required, as HostDeviceVector isn't header-only
template class HostDeviceVector<bst_float>;
template class HostDeviceVector<double>;
template class HostDeviceVector<GradientPair>;
template class HostDeviceVector<GradientPairPrecise>;
template class HostDeviceVector<int32_t>;   // bst_node_t
template class HostDeviceVector<uint8_t>;
template class HostDeviceVector<int8_t>;
template class HostDeviceVector<FeatureType>;
template class HostDeviceVector<Entry>;
template class HostDeviceVector<bst_idx_t>;
template class HostDeviceVector<uint32_t>;  // bst_feature_t

}  // namespace xgboost

#endif  // XGBOOST_USE_SYCL
