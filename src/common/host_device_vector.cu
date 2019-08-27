/*!
 * Copyright 2017 XGBoost contributors
 */

#include "./host_device_vector.h"
#include <thrust/fill.h>
#include <xgboost/data.h>
#include <algorithm>
#include <cstdint>
#include <mutex>
#include "./device_helpers.cuh"

namespace xgboost {

// the handler to call instead of cudaSetDevice; only used for testing
static void (*cudaSetDeviceHandler)(int) = nullptr;  // NOLINT

void SetCudaSetDeviceHandler(void (*handler)(int)) {
  cudaSetDeviceHandler = handler;
}

template <typename T>
class HostDeviceVectorImpl {
 public:
  HostDeviceVectorImpl(size_t size, T v, int device) : device_(device) {
    if (device >= 0) {
      gpu_access_ = GPUAccess::kWrite;
      SetDevice();
      data_d_.resize(size, v);
    } else {
      data_h_.resize(size, v);
    }
  }

  // Initializer can be std::vector<T> or std::initializer_list<T>
  template <class Initializer>
  HostDeviceVectorImpl(const Initializer& init, int device) : device_(device) {
    if (device >= 0) {
      gpu_access_ = GPUAccess::kWrite;
      LazyResizeDevice(init.size());
      Copy(init);
    } else {
      data_h_ = init;
    }
  }

  size_t Size() const { return HostCanRead() ? data_h_.size() : data_d_.size(); }

  int DeviceIdx() const { return device_; }

  T* DevicePointer() {
    LazySyncDevice(GPUAccess::kWrite);
    return data_d_.data().get();
  }

  const T* ConstDevicePointer() {
    LazySyncDevice(GPUAccess::kRead);
    return data_d_.data().get();
  }

  common::Span<T> DeviceSpan() {
    LazySyncDevice(GPUAccess::kWrite);
    return {data_d_.data().get(), static_cast<typename common::Span<T>::index_type>(Size())};
  }

  common::Span<const T> ConstDeviceSpan() {
    LazySyncDevice(GPUAccess::kRead);
    using SpanInd = typename common::Span<const T>::index_type;
    return {data_d_.data().get(), static_cast<SpanInd>(Size())};
  }

  thrust::device_ptr<T> tbegin() {  // NOLINT
    return thrust::device_ptr<T>(DevicePointer());
  }

  thrust::device_ptr<const T> tcbegin() {  // NOLINT
    return thrust::device_ptr<const T>(ConstDevicePointer());
  }

  thrust::device_ptr<T> tend() {  // NOLINT
    return tbegin() + Size();
  }

  thrust::device_ptr<const T> tcend() {  // NOLINT
    return tcbegin() + Size();
  }

  void Fill(T v) {  // NOLINT
    if (HostCanWrite()) {
      std::fill(data_h_.begin(), data_h_.end(), v);
    } else {
      gpu_access_ = GPUAccess::kWrite;
      SetDevice();
      thrust::fill(data_d_.begin(), data_d_.end(), v);
    }
  }

  void Copy(HostDeviceVectorImpl<T>* other) {
    CHECK_EQ(Size(), other->Size());
    // Data is on host.
    if (HostCanWrite() && other->HostCanWrite()) {
      std::copy(other->data_h_.begin(), other->data_h_.end(), data_h_.begin());
      return;
    }
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

  std::vector<T>& HostVector() {
    LazySyncHost(GPUAccess::kNone);
    return data_h_;
  }

  const std::vector<T>& ConstHostVector() {
    LazySyncHost(GPUAccess::kRead);
    return data_h_;
  }

  void SetDevice(int device) {
    if (device_ == device) { return; }
    if (device_ >= 0) {
      LazySyncHost(GPUAccess::kNone);
    }
    device_ = device;
    if (device_ >= 0) {
      LazyResizeDevice(data_h_.size());
    }
  }

  void Resize(size_t new_size, T v) {
    if (new_size == Size()) { return; }
    if (Size() == 0 && device_ >= 0) {
      // fast on-device resize
      gpu_access_ = GPUAccess::kWrite;
      SetDevice();
      data_d_.resize(new_size, v);
    } else {
      // resize on host
      LazySyncHost(GPUAccess::kNone);
      data_h_.resize(new_size, v);
    }
  }

  void LazySyncHost(GPUAccess access) {
    if (HostCanAccess(access)) { return; }
    if (HostCanRead()) {
      // data is present, just need to deny access to the device
      gpu_access_ = access;
      return;
    }
    gpu_access_ = access;
    if (data_h_.size() != data_d_.size()) { data_h_.resize(data_d_.size()); }
    SetDevice();
    dh::safe_cuda(cudaMemcpy(data_h_.data(),
                             data_d_.data().get(),
                             data_d_.size() * sizeof(T),
                             cudaMemcpyDeviceToHost));
  }

  void LazySyncDevice(GPUAccess access) {
    if (DeviceCanAccess(access)) { return; }
    if (DeviceCanRead()) {
      // deny read to the host
      gpu_access_ = access;
      return;
    }
    // data is on the host
    LazyResizeDevice(data_h_.size());
    SetDevice();
    dh::safe_cuda(cudaMemcpy(data_d_.data().get(),
                             data_h_.data(),
                             data_d_.size() * sizeof(T),
                             cudaMemcpyHostToDevice));
    gpu_access_ = access;
  }

  bool HostCanAccess(GPUAccess access) const { return gpu_access_ <= access; }
  bool HostCanRead() const { return HostCanAccess(GPUAccess::kRead); }
  bool HostCanWrite() const { return HostCanAccess(GPUAccess::kNone); }
  bool DeviceCanAccess(GPUAccess access) const { return gpu_access_ >= access; }
  bool DeviceCanRead() const { return DeviceCanAccess(GPUAccess::kRead); }
  bool DeviceCanWrite() const { return DeviceCanAccess(GPUAccess::kWrite); }

 private:
  int device_{-1};
  std::vector<T> data_h_{};
  dh::device_vector<T> data_d_{};
  GPUAccess gpu_access_{GPUAccess::kNone};

  void CopyToDevice(HostDeviceVectorImpl* other) {
    if (other->HostCanWrite()) {
      CopyToDevice(other->data_h_.data());
    } else {
      LazyResizeDevice(Size());
      gpu_access_ = GPUAccess::kWrite;
      SetDevice();
      dh::safe_cuda(cudaMemcpyAsync(data_d_.data().get(), other->data_d_.data().get(),
                                    data_d_.size() * sizeof(T), cudaMemcpyDefault));
    }
  }

  void CopyToDevice(const T* begin) {
    LazyResizeDevice(Size());
    gpu_access_ = GPUAccess::kWrite;
    SetDevice();
    dh::safe_cuda(cudaMemcpyAsync(data_d_.data().get(), begin,
                                  data_d_.size() * sizeof(T), cudaMemcpyDefault));
  }

  void LazyResizeDevice(size_t new_size) {
    if (new_size == data_d_.size()) { return; }
    SetDevice();
    data_d_.resize(new_size);
  }

  void SetDevice() {
    CHECK_GE(device_, 0);
    if (cudaSetDeviceHandler == nullptr) {
      dh::safe_cuda(cudaSetDevice(device_));
    } else {
      (*cudaSetDeviceHandler)(device_);
    }
  }
};

template<typename T>
HostDeviceVector<T>::HostDeviceVector(size_t size, T v, int device)
    : impl_(new HostDeviceVectorImpl<T>(size, v, device)) {}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(std::initializer_list<T> init, int device)
    : impl_(new HostDeviceVectorImpl<T>(init, device)) {}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(const std::vector<T>& init, int device)
    : impl_(new HostDeviceVectorImpl<T>(init, device)) {}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(const HostDeviceVector<T>& other)
    : impl_(new HostDeviceVectorImpl<T>(*other.impl_)) {}

template <typename T>
HostDeviceVector<T>& HostDeviceVector<T>::operator=(const HostDeviceVector<T>& other) {
  if (this == &other) { return *this; }

  std::unique_ptr<HostDeviceVectorImpl<T>> newImpl(new HostDeviceVectorImpl<T>(*other.impl_));
  delete impl_;
  impl_ = newImpl.release();
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
int HostDeviceVector<T>::DeviceIdx() const { return impl_->DeviceIdx(); }

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
thrust::device_ptr<T> HostDeviceVector<T>::tbegin() {  // NOLINT
  return impl_->tbegin();
}

template <typename T>
thrust::device_ptr<const T> HostDeviceVector<T>::tcbegin() const {  // NOLINT
  return impl_->tcbegin();
}

template <typename T>
thrust::device_ptr<T> HostDeviceVector<T>::tend() {  // NOLINT
  return impl_->tend();
}

template <typename T>
thrust::device_ptr<const T> HostDeviceVector<T>::tcend() const {  // NOLINT
  return impl_->tcend();
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
std::vector<T>& HostDeviceVector<T>::HostVector() { return impl_->HostVector(); }

template <typename T>
const std::vector<T>& HostDeviceVector<T>::ConstHostVector() const {
  return impl_->ConstHostVector();
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
void HostDeviceVector<T>::SetDevice(int device) const {
  impl_->SetDevice(device);
}

template <typename T>
void HostDeviceVector<T>::Resize(size_t new_size, T v) {
  impl_->Resize(new_size, v);
}

// explicit instantiations are required, as HostDeviceVector isn't header-only
template class HostDeviceVector<bst_float>;
template class HostDeviceVector<GradientPair>;
template class HostDeviceVector<int>;
template class HostDeviceVector<Entry>;
template class HostDeviceVector<size_t>;

}  // namespace xgboost
