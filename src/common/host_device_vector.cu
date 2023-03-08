/**
 * Copyright 2017-2023 by XGBoost contributors
 */
#include <thrust/fill.h>
#include <thrust/device_ptr.h>

#include <algorithm>
#include <cstdint>
#include <mutex>

#include "xgboost/data.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/tree_model.h"
#include "device_helpers.cuh"

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
      data_d_->resize(size, v);
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

  HostDeviceVectorImpl(HostDeviceVectorImpl<T>&& that) :
    device_{that.device_},
    data_h_{std::move(that.data_h_)},
    data_d_{std::move(that.data_d_)},
    gpu_access_{that.gpu_access_} {}

  ~HostDeviceVectorImpl() {
    if (device_ >= 0) {
      SetDevice();
    }
  }

  size_t Size() const {
    return HostCanRead() ? data_h_.size() : data_d_ ? data_d_->size() : 0;
  }

  int DeviceIdx() const { return device_; }

  T* DevicePointer() {
    LazySyncDevice(GPUAccess::kWrite);
    return data_d_->data().get();
  }

  const T* ConstDevicePointer() {
    LazySyncDevice(GPUAccess::kRead);
    return data_d_->data().get();
  }

  common::Span<T> DeviceSpan() {
    LazySyncDevice(GPUAccess::kWrite);
    return {data_d_->data().get(), Size()};
  }

  common::Span<const T> ConstDeviceSpan() {
    LazySyncDevice(GPUAccess::kRead);
    return {data_d_->data().get(), Size()};
  }

  void Fill(T v) {  // NOLINT
    if (HostCanWrite()) {
      std::fill(data_h_.begin(), data_h_.end(), v);
    } else {
      gpu_access_ = GPUAccess::kWrite;
      SetDevice();
      auto s_data = dh::ToSpan(*data_d_);
      dh::LaunchN(data_d_->size(),
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
    this->Resize(ori_size + other->Size(), T());
    if (HostCanWrite() && other->HostCanRead()) {
      auto& h_vec = this->HostVector();
      auto& other_vec = other->HostVector();
      CHECK_EQ(h_vec.size(), ori_size + other->Size());
      std::copy(other_vec.cbegin(), other_vec.cend(), h_vec.begin() + ori_size);
    } else {
      auto ptr = other->ConstDevicePointer();
      SetDevice();
      CHECK_EQ(this->DeviceIdx(), other->DeviceIdx());
      dh::safe_cuda(cudaMemcpyAsync(this->DevicePointer() + ori_size,
                                    ptr,
                                    other->Size() * sizeof(T),
                                    cudaMemcpyDeviceToDevice));
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

    if (device_ >= 0 && device >= 0) {
      CHECK_EQ(device_, device) << "New device ordinal is different from previous one.";
    }
    device_ = device;
    if (device_ >= 0) {
      LazyResizeDevice(data_h_.size());
    }
  }

  void Resize(size_t new_size, T v) {
    if (new_size == Size()) { return; }
    if ((Size() == 0 && device_ >= 0) || (DeviceCanWrite() && device_ >= 0)) {
      // fast on-device resize
      gpu_access_ = GPUAccess::kWrite;
      SetDevice();
      data_d_->resize(new_size, v);
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
    if (data_h_.size() != data_d_->size()) { data_h_.resize(data_d_->size()); }
    SetDevice();
    dh::safe_cuda(cudaMemcpy(data_h_.data(),
                             data_d_->data().get(),
                             data_d_->size() * sizeof(T),
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
    dh::safe_cuda(cudaMemcpyAsync(data_d_->data().get(),
                                  data_h_.data(),
                                  data_d_->size() * sizeof(T),
                                  cudaMemcpyHostToDevice));
    gpu_access_ = access;
  }

  bool HostCanAccess(GPUAccess access) const { return gpu_access_ <= access; }
  bool HostCanRead() const { return HostCanAccess(GPUAccess::kRead); }
  bool HostCanWrite() const { return HostCanAccess(GPUAccess::kNone); }
  bool DeviceCanAccess(GPUAccess access) const { return gpu_access_ >= access; }
  bool DeviceCanRead() const { return DeviceCanAccess(GPUAccess::kRead); }
  bool DeviceCanWrite() const { return DeviceCanAccess(GPUAccess::kWrite); }
  GPUAccess Access() const { return gpu_access_; }

 private:
  int device_{-1};
  std::vector<T> data_h_{};
  std::unique_ptr<dh::device_vector<T>> data_d_{};
  GPUAccess gpu_access_{GPUAccess::kNone};

  void CopyToDevice(HostDeviceVectorImpl* other) {
    if (other->HostCanWrite()) {
      CopyToDevice(other->data_h_.data());
    } else {
      LazyResizeDevice(Size());
      gpu_access_ = GPUAccess::kWrite;
      SetDevice();
      dh::safe_cuda(cudaMemcpyAsync(data_d_->data().get(), other->data_d_->data().get(),
                                    data_d_->size() * sizeof(T), cudaMemcpyDefault));
    }
  }

  void CopyToDevice(const T* begin) {
    LazyResizeDevice(Size());
    gpu_access_ = GPUAccess::kWrite;
    SetDevice();
    dh::safe_cuda(cudaMemcpyAsync(data_d_->data().get(), begin,
                                  data_d_->size() * sizeof(T), cudaMemcpyDefault));
  }

  void LazyResizeDevice(size_t new_size) {
    if (data_d_ && new_size == data_d_->size()) { return; }
    SetDevice();
    data_d_->resize(new_size);
  }

  void SetDevice() {
    CHECK_GE(device_, 0);
    if (cudaSetDeviceHandler == nullptr) {
      dh::safe_cuda(cudaSetDevice(device_));
    } else {
      (*cudaSetDeviceHandler)(device_);
    }

    if (!data_d_) {
      data_d_.reset(new dh::device_vector<T>);
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
void HostDeviceVector<T>::SetDevice(int device) const {
  impl_->SetDevice(device);
}

template <typename T>
void HostDeviceVector<T>::Resize(size_t new_size, T v) {
  impl_->Resize(new_size, v);
}

// explicit instantiations are required, as HostDeviceVector isn't header-only
template class HostDeviceVector<bst_float>;
template class HostDeviceVector<double>;
template class HostDeviceVector<GradientPair>;
template class HostDeviceVector<GradientPairPrecise>;
template class HostDeviceVector<int32_t>;   // bst_node_t
template class HostDeviceVector<uint8_t>;
template class HostDeviceVector<FeatureType>;
template class HostDeviceVector<Entry>;
template class HostDeviceVector<uint64_t>;  // bst_row_t
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
