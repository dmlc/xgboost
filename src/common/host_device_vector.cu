/**
 * Copyright 2017-2024, XGBoost contributors
 */
#include <thrust/fill.h>

#include <algorithm>
#include <cstddef>  // for size_t
#include <cstdint>

#include "device_helpers.cuh"
#include "device_vector.cuh"  // for DeviceUVector
#include "xgboost/data.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/tree_model.h"  // for RegTree

namespace xgboost {

// the handler to call instead of cudaSetDevice; only used for testing
static void (*cudaSetDeviceHandler)(int) = nullptr;  // NOLINT

void SetCudaSetDeviceHandler(void (*handler)(int)) {
  cudaSetDeviceHandler = handler;
}

template <typename T>
class HostDeviceVectorImpl {
 public:
  HostDeviceVectorImpl(size_t size, T v, DeviceOrd device) : device_(device) {
    if (device.IsCUDA()) {
      gpu_access_ = GPUAccess::kWrite;
      SetDevice();
      data_d_->Resize(size, v);
    } else {
      data_h_.resize(size, v);
    }
  }

  // Initializer can be std::vector<T> or std::initializer_list<T>
  template <class Initializer>
  HostDeviceVectorImpl(const Initializer& init, DeviceOrd device) : device_(device) {
    if (device.IsCUDA()) {
      gpu_access_ = GPUAccess::kWrite;
      LazyResizeDevice(init.size());
      Copy(init);
    } else {
      data_h_ = init;
    }
  }

  HostDeviceVectorImpl(HostDeviceVectorImpl<T>&& that) :
    device_{that.device_},
    data_h_{std::move(that.data_h_)},
    data_d_{std::move(that.data_d_)},
    gpu_access_{that.gpu_access_} {}

  ~HostDeviceVectorImpl() {
    if (device_.IsCUDA()) {
      SetDevice();
    }
  }

  [[nodiscard]] size_t Size() const {
    return HostCanRead() ? data_h_.size() : data_d_ ? data_d_->size() : 0;
  }

  [[nodiscard]] DeviceOrd Device() const { return device_; }

  T* DevicePointer() {
    LazySyncDevice(GPUAccess::kWrite);
    return thrust::raw_pointer_cast(data_d_->data());
  }

  const T* ConstDevicePointer() {
    LazySyncDevice(GPUAccess::kRead);
    return thrust::raw_pointer_cast(data_d_->data());
  }

  common::Span<T> DeviceSpan() {
    LazySyncDevice(GPUAccess::kWrite);
    return {this->DevicePointer(), Size()};
  }

  common::Span<const T> ConstDeviceSpan() {
    LazySyncDevice(GPUAccess::kRead);
    return {this->ConstDevicePointer(), Size()};
  }

  void Fill(T v) {  // NOLINT
    if (HostCanWrite()) {
      std::fill(data_h_.begin(), data_h_.end(), v);
    } else {
      gpu_access_ = GPUAccess::kWrite;
      SetDevice();
      auto s_data = dh::ToSpan(*data_d_);
      dh::LaunchN(data_d_->size(), dh::DefaultStream(),
                  [=] XGBOOST_DEVICE(size_t i) { s_data[i] = v; });
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
      dh::safe_cuda(cudaMemcpyAsync(this->DevicePointer() + ori_size, ptr,
                                    other->Size() * sizeof(T), cudaMemcpyDeviceToDevice,
                                    dh::DefaultStream()));
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

  void SetDevice(DeviceOrd device) {
    if (device_ == device) { return; }
    if (device_.IsCUDA()) {
      LazySyncHost(GPUAccess::kNone);
    }

    if (device_.IsCUDA() && device.IsCUDA()) {
      CHECK_EQ(device_.ordinal, device.ordinal)
          << "New device ordinal is different from previous one.";
    }
    device_ = device;
    if (device_.IsCUDA()) {
      LazyResizeDevice(data_h_.size());
    }
  }

  template <typename... U>
  auto Resize(std::size_t new_size, U&&... args) {
    if (new_size == Size()) {
      return;
    }
    if ((Size() == 0 && device_.IsCUDA()) || (DeviceCanWrite() && device_.IsCUDA())) {
      // fast on-device resize
      gpu_access_ = GPUAccess::kWrite;
      SetDevice();
      auto old_size = data_d_->size();
      data_d_->Resize(new_size, std::forward<U>(args)...);
    } else {
      // resize on host
      LazySyncHost(GPUAccess::kNone);
      auto old_size = data_h_.size();
      data_h_.resize(new_size, std::forward<U>(args)...);
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
    if (data_h_.size() != data_d_->size()) { data_h_.resize(data_d_->size()); }
    SetDevice();
    dh::safe_cuda(cudaMemcpy(data_h_.data(), thrust::raw_pointer_cast(data_d_->data()),
                             data_d_->size() * sizeof(T), cudaMemcpyDeviceToHost));
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
    dh::safe_cuda(cudaMemcpyAsync(thrust::raw_pointer_cast(data_d_->data()), data_h_.data(),
                                  data_d_->size() * sizeof(T), cudaMemcpyHostToDevice,
                                  dh::DefaultStream()));
    gpu_access_ = access;
  }

  [[nodiscard]] bool HostCanAccess(GPUAccess access) const { return gpu_access_ <= access; }
  [[nodiscard]] bool HostCanRead() const { return HostCanAccess(GPUAccess::kRead); }
  [[nodiscard]] bool HostCanWrite() const { return HostCanAccess(GPUAccess::kNone); }
  [[nodiscard]] bool DeviceCanAccess(GPUAccess access) const { return gpu_access_ >= access; }
  [[nodiscard]] bool DeviceCanRead() const { return DeviceCanAccess(GPUAccess::kRead); }
  [[nodiscard]] bool DeviceCanWrite() const { return DeviceCanAccess(GPUAccess::kWrite); }
  [[nodiscard]] GPUAccess Access() const { return gpu_access_; }

 private:
  DeviceOrd device_{DeviceOrd::CPU()};
  std::vector<T> data_h_{};
  std::unique_ptr<dh::DeviceUVector<T>> data_d_{};
  GPUAccess gpu_access_{GPUAccess::kNone};

  void CopyToDevice(HostDeviceVectorImpl* other) {
    if (other->HostCanWrite()) {
      CopyToDevice(other->data_h_.data());
    } else {
      LazyResizeDevice(Size());
      gpu_access_ = GPUAccess::kWrite;
      SetDevice();
      dh::safe_cuda(cudaMemcpyAsync(thrust::raw_pointer_cast(data_d_->data()),
                                    thrust::raw_pointer_cast(other->data_d_->data()),
                                    data_d_->size() * sizeof(T), cudaMemcpyDefault,
                                    dh::DefaultStream()));
    }
  }

  void CopyToDevice(const T* begin) {
    LazyResizeDevice(Size());
    gpu_access_ = GPUAccess::kWrite;
    SetDevice();
    dh::safe_cuda(cudaMemcpyAsync(thrust::raw_pointer_cast(data_d_->data()), begin,
                                  data_d_->size() * sizeof(T), cudaMemcpyDefault,
                                  dh::DefaultStream()));
  }

  void LazyResizeDevice(size_t new_size) {
    if (data_d_ && new_size == data_d_->size()) { return; }
    SetDevice();
    data_d_->Resize(new_size);
  }

  void SetDevice() {
    CHECK_GE(device_.ordinal, 0);
    if (cudaSetDeviceHandler == nullptr) {
      dh::safe_cuda(cudaSetDevice(device_.ordinal));
    } else {
      (*cudaSetDeviceHandler)(device_.ordinal);
    }

    if (!data_d_) {
      data_d_.reset(new dh::DeviceUVector<T>{});
    }
  }
};

template<typename T>
HostDeviceVector<T>::HostDeviceVector(size_t size, T v, DeviceOrd device)
    : impl_(new HostDeviceVectorImpl<T>(size, v, device)) {}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(std::initializer_list<T> init, DeviceOrd device)
    : impl_(new HostDeviceVectorImpl<T>(init, device)) {}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(const std::vector<T>& init, DeviceOrd device)
    : impl_(new HostDeviceVectorImpl<T>(init, device)) {}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(HostDeviceVector<T>&& other)
    : impl_(new HostDeviceVectorImpl<T>(std::move(*other.impl_))) {}

template <typename T>
HostDeviceVector<T>& HostDeviceVector<T>::operator=(HostDeviceVector<T>&& other) {
  if (this == &other) { return *this; }

  std::unique_ptr<HostDeviceVectorImpl<T>> new_impl(
      new HostDeviceVectorImpl<T>(std::move(*other.impl_)));
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
GPUAccess HostDeviceVector<T>::DeviceAccess() const {
  return impl_->Access();
}

template <typename T>
void HostDeviceVector<T>::SetDevice(DeviceOrd device) const {
  impl_->SetDevice(device);
}

template <typename T>
void HostDeviceVector<T>::Resize(std::size_t new_size) {
  impl_->Resize(new_size);
}

template <typename T>
void HostDeviceVector<T>::Resize(std::size_t new_size, T v) {
  impl_->Resize(new_size, v);
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
template class HostDeviceVector<RegTree::Node>;
template class HostDeviceVector<RegTree::CategoricalSplitMatrix::Segment>;
template class HostDeviceVector<RTreeNodeStat>;

#if defined(__APPLE__)
/*
 * On OSX:
 *
 * typedef unsigned int         uint32_t;
 * typedef unsigned long long   uint64_t;
 * typedef unsigned long       __darwin_size_t;
 */
template class HostDeviceVector<std::size_t>;
#endif  // defined(__APPLE__)
}  // namespace xgboost
