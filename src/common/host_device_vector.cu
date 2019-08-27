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

// wrapper over access with useful methods
class Permissions {
  GPUAccess access_;
  explicit Permissions(GPUAccess access) : access_{access} {}

 public:
  Permissions() : access_{GPUAccess::kNone} {}
  explicit Permissions(bool perm)
    : access_(perm ? GPUAccess::kWrite : GPUAccess::kNone) {}

  bool CanRead() const { return access_ >= kRead; }
  bool CanWrite() const { return access_ == kWrite; }
  bool CanAccess(GPUAccess access) const { return access_ >= access; }
  void Grant(GPUAccess access) { access_ = std::max(access_, access); }
  void DenyComplementary(GPUAccess compl_access) {
    access_ = std::min(access_, GPUAccess::kWrite - compl_access);
  }
  Permissions Complementary() const {
    return Permissions(GPUAccess::kWrite - access_);
  }
};

template <typename T>
class HostDeviceVectorImpl {
 public:
  HostDeviceVectorImpl(size_t size, T v, int device) : device_(device), perm_h_(device < 0) {
    if (device >= 0) {
      SetDevice();
      data_d_.resize(size, v);
    } else {
      data_h_.resize(size, v);
    }
  }

  // required, as a new std::mutex has to be created
  HostDeviceVectorImpl(const HostDeviceVectorImpl<T>& other)
      : device_(other.device_), data_h_(other.data_h_), perm_h_(other.perm_h_), mutex_() {
    if (device_ >= 0) {
      SetDevice();
      data_d_ = other.data_d_;
    }
  }

  // Initializer can be std::vector<T> or std::initializer_list<T>
  template <class Initializer>
  HostDeviceVectorImpl(const Initializer& init, int device) : device_(device), perm_h_(device < 0) {
    if (device >= 0) {
      LazyResizeDevice(init.size());
      Copy(init);
    } else {
      data_h_ = init;
    }
  }

  ~HostDeviceVectorImpl() {
    if (device_ >= 0) {
      SetDevice();
    }
  }

  size_t Size() const { return perm_h_.CanRead() ? data_h_.size() : data_d_.size(); }

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
    if (perm_h_.CanWrite()) {
      std::fill(data_h_.begin(), data_h_.end(), v);
    } else {
      SetDevice();
      thrust::fill(data_d_.begin(), data_d_.end(), v);
    }
  }

  void Copy(HostDeviceVectorImpl<T>* other) {
    CHECK_EQ(Size(), other->Size());
    // Data is on host.
    if (perm_h_.CanWrite() && other->perm_h_.CanWrite()) {
      std::copy(other->data_h_.begin(), other->data_h_.end(), data_h_.begin());
      return;
    }
    // Data is on device;
    other->SetDevice(device_);
    DeviceCopy(other);
  }

  void Copy(const std::vector<T>& other) {
    CHECK_EQ(Size(), other.size());
    if (perm_h_.CanWrite()) {
      std::copy(other.begin(), other.end(), data_h_.begin());
    } else {
      DeviceCopy(other.data());
    }
  }

  void Copy(std::initializer_list<T> other) {
    CHECK_EQ(Size(), other.size());
    if (perm_h_.CanWrite()) {
      std::copy(other.begin(), other.end(), data_h_.begin());
    } else {
      DeviceCopy(other.begin());
    }
  }

  std::vector<T>& HostVector() {
    LazySyncHost(GPUAccess::kWrite);
    return data_h_;
  }

  const std::vector<T>& ConstHostVector() {
    LazySyncHost(GPUAccess::kRead);
    return data_h_;
  }

  void SetDevice(int device) {
    if (device_ == device) { return; }
    if (device_ >= 0) {
      LazySyncHost(GPUAccess::kWrite);
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
      perm_h_ = Permissions(false);
      SetDevice();
      data_d_.resize(new_size, v);
    } else {
      // resize on host
      LazySyncHost(GPUAccess::kWrite);
      data_h_.resize(new_size, v);
    }
  }

  void LazySyncHost(GPUAccess access) {
    if (perm_h_.CanAccess(access)) { return; }
    if (perm_h_.CanRead()) {
      // data is present, just need to deny access to the device
      perm_h_.Grant(access);
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (data_h_.size() != data_d_.size()) { data_h_.resize(data_d_.size()); }
    SetDevice();
    dh::safe_cuda(cudaMemcpy(data_h_.data(),
                             data_d_.data().get(),
                             data_d_.size() * sizeof(T),
                             cudaMemcpyDeviceToHost));
    perm_h_.Grant(access);
  }

  void LazySyncDevice(GPUAccess access) {
    if (DevicePerm().CanAccess(access)) { return; }
    if (DevicePerm().CanRead()) {
      // deny read to the host
      std::lock_guard<std::mutex> lock(mutex_);
      perm_h_.DenyComplementary(access);
      return;
    }
    // data is on the host
    LazyResizeDevice(data_h_.size());
    SetDevice();
    dh::safe_cuda(cudaMemcpy(data_d_.data().get(),
                             data_h_.data(),
                             data_d_.size() * sizeof(T),
                             cudaMemcpyHostToDevice));

    std::lock_guard<std::mutex> lock(mutex_);
    perm_h_.DenyComplementary(access);
  }

  bool HostCanAccess(GPUAccess access) { return perm_h_.CanAccess(access); }
  bool DeviceCanAccess(GPUAccess access) { return DevicePerm().CanAccess(access); }

 private:
  int device_{-1};
  std::vector<T> data_h_{};
  dh::device_vector<T> data_d_{};
  Permissions perm_h_{false};
  // protects size_d_ and perm_h_ when updated from multiple threads
  std::mutex mutex_{};

  void DeviceCopy(HostDeviceVectorImpl* other) {
    if (other->perm_h_.CanWrite()) {
      DeviceCopy(other->data_h_.data());
    } else {
      LazyResizeDevice(Size());
      std::lock_guard<std::mutex> lock(mutex_);
      perm_h_.DenyComplementary(GPUAccess::kWrite);
      SetDevice();
      dh::safe_cuda(cudaMemcpyAsync(data_d_.data().get(), other->data_d_.data().get(),
                                    data_d_.size() * sizeof(T), cudaMemcpyDefault));
    }
  }

  void DeviceCopy(const T* begin) {
    LazyResizeDevice(Size());
    std::lock_guard<std::mutex> lock(mutex_);
    perm_h_.DenyComplementary(GPUAccess::kWrite);
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

  Permissions DevicePerm() const { return perm_h_.Complementary(); }
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
bool HostDeviceVector<T>::HostCanAccess(GPUAccess access) const {
  return impl_->HostCanAccess(access);
}

template <typename T>
bool HostDeviceVector<T>::DeviceCanAccess(GPUAccess access) const {
  return impl_->DeviceCanAccess(access);
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
