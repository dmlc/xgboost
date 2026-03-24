/**
 * Copyright 2017-2026, XGBoost contributors
 */
#include <thrust/fill.h>

#include <algorithm>
#include <cstddef>  // for size_t
#include <cstdint>

#include "cuda_context.cuh"  // for CUDAContext
#include "cuda_stream.h"     // for DefaultStream
#include "device_helpers.cuh"
#include "device_vector.cuh"  // for DeviceUVector
#include "xgboost/data.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/tree_model.h"  // for RegTree

namespace xgboost {

// the handler to call instead of cudaSetDevice; only used for testing
static void (*cudaSetDeviceHandler)(int) = nullptr;  // NOLINT

void SetCudaSetDeviceHandler(void (*handler)(int)) { cudaSetDeviceHandler = handler; }

namespace {
curt::StreamRef GetStream(CUDAContext const* ctx) {
  return ctx ? ctx->Stream() : curt::DefaultStream();
}

CUDAContext const* GetCUDACtx(Context const* ctx) {
  return ctx && ctx->IsCUDA() ? ctx->CUDACtx() : nullptr;
}
}  // namespace

template <typename T>
class HostDeviceVectorImpl {
 public:
  HostDeviceVectorImpl(CUDAContext const* ctx, size_t size, T v, DeviceOrd device)
      : device_(device) {
    if (device.IsCUDA()) {
      gpu_access_ = GPUAccess::kWrite;
      SetDevice();
      data_d_->resize(size, v, GetStream(ctx));
    } else {
      data_h_.resize(size, v);
    }
  }

  // Initializer can be std::vector<T> or std::initializer_list<T>
  template <class Initializer>
  HostDeviceVectorImpl(CUDAContext const* ctx, const Initializer& init, DeviceOrd device)
      : device_(device) {
    if (device.IsCUDA()) {
      gpu_access_ = GPUAccess::kWrite;
      LazyResizeDevice(init.size(), ctx);
      Copy(ctx, init);
    } else {
      data_h_ = init;
    }
  }

  HostDeviceVectorImpl(HostDeviceVectorImpl<T>&& that)
      : device_{that.device_},
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

  T* DevicePointer(CUDAContext const* ctx) {
    LazySyncDevice(ctx, GPUAccess::kWrite);
    return data_d_->data();
  }

  const T* ConstDevicePointer(CUDAContext const* ctx) {
    LazySyncDevice(ctx, GPUAccess::kRead);
    return data_d_->data();
  }

  common::Span<T> DeviceSpan(CUDAContext const* ctx) {
    LazySyncDevice(ctx, GPUAccess::kWrite);
    return {this->DevicePointer(ctx), Size()};
  }

  common::Span<const T> ConstDeviceSpan(CUDAContext const* ctx) {
    LazySyncDevice(ctx, GPUAccess::kRead);
    return {this->ConstDevicePointer(ctx), Size()};
  }

  void Fill(T v, CUDAContext const* ctx) {  // NOLINT
    if (HostCanWrite()) {
      std::fill(data_h_.begin(), data_h_.end(), v);
    } else {
      gpu_access_ = GPUAccess::kWrite;
      SetDevice();
      auto s_data = dh::ToSpan(*data_d_);
      dh::LaunchN(data_d_->size(), GetStream(ctx), [=] XGBOOST_DEVICE(size_t i) { s_data[i] = v; });
    }
  }

  void Copy(CUDAContext const* ctx, HostDeviceVectorImpl<T>* other) {
    CHECK_EQ(Size(), other->Size());
    SetDevice(other->device_, ctx);
    // Data is on host.
    if (HostCanWrite() && other->HostCanWrite()) {
      std::copy(other->data_h_.begin(), other->data_h_.end(), data_h_.begin());
      return;
    }
    SetDevice();
    CopyToDevice(ctx, other);
  }

  void Copy(CUDAContext const* ctx, const std::vector<T>& other) {
    CHECK_EQ(Size(), other.size());
    if (HostCanWrite()) {
      std::copy(other.begin(), other.end(), data_h_.begin());
    } else {
      CopyToDevice(ctx, other.data());
    }
  }

  void Copy(CUDAContext const* ctx, std::initializer_list<T> other) {
    CHECK_EQ(Size(), other.size());
    if (HostCanWrite()) {
      std::copy(other.begin(), other.end(), data_h_.begin());
    } else {
      CopyToDevice(ctx, other.begin());
    }
  }

  void Extend(CUDAContext const* ctx, HostDeviceVectorImpl* other) {
    auto ori_size = this->Size();
    this->Resize(ctx, ori_size + other->Size(), T{});
    if (HostCanWrite() && other->HostCanRead()) {
      auto& h_vec = this->HostVector(ctx);
      auto& other_vec = other->HostVector(ctx);
      CHECK_EQ(h_vec.size(), ori_size + other->Size());
      std::copy(other_vec.cbegin(), other_vec.cend(), h_vec.begin() + ori_size);
    } else {
      auto ptr = other->ConstDevicePointer(ctx);
      SetDevice();
      CHECK_EQ(this->Device(), other->Device());
      dh::safe_cuda(cudaMemcpyAsync(this->DevicePointer(ctx) + ori_size, ptr,
                                    other->Size() * sizeof(T), cudaMemcpyDeviceToDevice,
                                    GetStream(ctx)));
    }
  }

  std::vector<T>& HostVector(CUDAContext const* ctx) {
    LazySyncHost(ctx, GPUAccess::kNone);
    return data_h_;
  }

  const std::vector<T>& ConstHostVector(CUDAContext const* ctx) {
    LazySyncHost(ctx, GPUAccess::kRead);
    return data_h_;
  }

  void SetDevice(DeviceOrd device, CUDAContext const* ctx) {
    if (device_ == device) {
      return;
    }
    if (device_.IsCUDA()) {
      LazySyncHost(ctx, GPUAccess::kNone);
    }

    if (device_.IsCUDA() && device.IsCUDA()) {
      CHECK_EQ(device_.ordinal, device.ordinal)
          << "New device ordinal is different from previous one.";
    }
    device_ = device;
    if (device_.IsCUDA()) {
      LazyResizeDevice(data_h_.size(), ctx);
    }
  }

  template <typename... U>
  auto Resize(CUDAContext const* ctx, std::size_t new_size, U&&... args) {
    if (new_size == Size()) {
      return;
    }
    if ((Size() == 0 && device_.IsCUDA()) || (DeviceCanWrite() && device_.IsCUDA())) {
      // fast on-device resize
      gpu_access_ = GPUAccess::kWrite;
      SetDevice();
      auto old_size = data_d_->size();
      data_d_->resize(new_size, std::forward<U>(args)..., GetStream(ctx));
    } else {
      // resize on host
      LazySyncHost(ctx, GPUAccess::kNone);
      auto old_size = data_h_.size();
      data_h_.resize(new_size, std::forward<U>(args)...);
    }
  }

  void LazySyncHost(CUDAContext const* ctx, GPUAccess access) {
    if (HostCanAccess(access)) {
      return;
    }
    if (HostCanRead()) {
      gpu_access_ = access;
      return;
    }
    gpu_access_ = access;
    if (data_h_.size() != data_d_->size()) {
      data_h_.resize(data_d_->size());
    }
    SetDevice();
    auto stream = GetStream(ctx);
    dh::safe_cuda(cudaMemcpyAsync(data_h_.data(), data_d_->data(), data_d_->size() * sizeof(T),
                                  cudaMemcpyDeviceToHost, stream));
    dh::safe_cuda(cudaStreamSynchronize(stream));
  }

  void LazySyncDevice(CUDAContext const* ctx, GPUAccess access) {
    if (DeviceCanAccess(access)) {
      return;
    }
    if (DeviceCanRead()) {
      gpu_access_ = access;
      return;
    }
    // data is on the host
    LazyResizeDevice(data_h_.size(), ctx);
    SetDevice();
    dh::safe_cuda(cudaMemcpyAsync(data_d_->data(), data_h_.data(), data_d_->size() * sizeof(T),
                                  cudaMemcpyHostToDevice, GetStream(ctx)));
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

  void CopyToDevice(CUDAContext const* ctx, HostDeviceVectorImpl* other) {
    if (other->HostCanWrite()) {
      CopyToDevice(ctx, other->data_h_.data());
    } else {
      LazyResizeDevice(Size(), ctx);
      gpu_access_ = GPUAccess::kWrite;
      SetDevice();
      dh::safe_cuda(cudaMemcpyAsync(data_d_->data(), other->data_d_->data(),
                                    data_d_->size() * sizeof(T), cudaMemcpyDefault,
                                    GetStream(ctx)));
    }
  }

  void CopyToDevice(CUDAContext const* ctx, const T* begin) {
    LazyResizeDevice(Size(), ctx);
    gpu_access_ = GPUAccess::kWrite;
    SetDevice();
    dh::safe_cuda(cudaMemcpyAsync(data_d_->data(), begin, data_d_->size() * sizeof(T),
                                  cudaMemcpyDefault, GetStream(ctx)));
  }

  void LazyResizeDevice(size_t new_size, CUDAContext const* ctx) {
    if (data_d_ && new_size == data_d_->size()) {
      return;
    }
    SetDevice();
    data_d_->resize(new_size, GetStream(ctx));
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

template <typename T>
HostDeviceVector<T>::HostDeviceVector(size_t size, T v, DeviceOrd device, Context const* ctx)
    : impl_(new HostDeviceVectorImpl<T>(GetCUDACtx(ctx), size, v, device)) {}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(std::initializer_list<T> init, DeviceOrd device,
                                      Context const* ctx)
    : impl_(new HostDeviceVectorImpl<T>(GetCUDACtx(ctx), init, device)) {}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(const std::vector<T>& init, DeviceOrd device,
                                      Context const* ctx)
    : impl_(new HostDeviceVectorImpl<T>(GetCUDACtx(ctx), init, device)) {}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(HostDeviceVector<T>&& other)
    : impl_(new HostDeviceVectorImpl<T>(std::move(*other.impl_))) {}

template <typename T>
HostDeviceVector<T>& HostDeviceVector<T>::operator=(HostDeviceVector<T>&& other) {
  if (this == &other) {
    return *this;
  }

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
size_t HostDeviceVector<T>::Size() const {
  return impl_->Size();
}

template <typename T>
DeviceOrd HostDeviceVector<T>::Device() const {
  return impl_->Device();
}

template <typename T>
T* HostDeviceVector<T>::DevicePointer(Context const* ctx) {
  return impl_->DevicePointer(GetCUDACtx(ctx));
}

template <typename T>
const T* HostDeviceVector<T>::ConstDevicePointer(Context const* ctx) const {
  return impl_->ConstDevicePointer(GetCUDACtx(ctx));
}

template <typename T>
common::Span<T> HostDeviceVector<T>::DeviceSpan(Context const* ctx) {
  return impl_->DeviceSpan(GetCUDACtx(ctx));
}

template <typename T>
common::Span<const T> HostDeviceVector<T>::ConstDeviceSpan(Context const* ctx) const {
  return impl_->ConstDeviceSpan(GetCUDACtx(ctx));
}

template <typename T>
void HostDeviceVector<T>::Fill(T v, Context const* ctx) {
  impl_->Fill(v, GetCUDACtx(ctx));
}

template <typename T>
void HostDeviceVector<T>::Copy(const HostDeviceVector<T>& other, Context const* ctx) {
  impl_->Copy(GetCUDACtx(ctx), other.impl_);
}

template <typename T>
void HostDeviceVector<T>::Copy(const std::vector<T>& other, Context const* ctx) {
  impl_->Copy(GetCUDACtx(ctx), other);
}

template <typename T>
void HostDeviceVector<T>::Copy(std::initializer_list<T> other, Context const* ctx) {
  impl_->Copy(GetCUDACtx(ctx), other);
}

template <typename T>
void HostDeviceVector<T>::Extend(HostDeviceVector const& other, Context const* ctx) {
  impl_->Extend(GetCUDACtx(ctx), other.impl_);
}

template <typename T>
std::vector<T>& HostDeviceVector<T>::HostVector(Context const* ctx) {
  return impl_->HostVector(GetCUDACtx(ctx));
}

template <typename T>
const std::vector<T>& HostDeviceVector<T>::ConstHostVector(Context const* ctx) const {
  return impl_->ConstHostVector(GetCUDACtx(ctx));
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
void HostDeviceVector<T>::SetDevice(DeviceOrd device, Context const* ctx) const {
  impl_->SetDevice(device, GetCUDACtx(ctx));
}

template <typename T>
void HostDeviceVector<T>::Resize(std::size_t new_size) {
  impl_->Resize(nullptr, new_size);
}

template <typename T>
void HostDeviceVector<T>::Resize(Context const* ctx, std::size_t new_size) {
  impl_->Resize(GetCUDACtx(ctx), new_size);
}

template <typename T>
void HostDeviceVector<T>::Resize(Context const* ctx, std::size_t new_size, T v) {
  impl_->Resize(GetCUDACtx(ctx), new_size, v);
}

// explicit instantiations are required, as HostDeviceVector isn't header-only
template class HostDeviceVector<bst_float>;
template class HostDeviceVector<double>;
template class HostDeviceVector<GradientPair>;
template class HostDeviceVector<GradientPairPrecise>;
template class HostDeviceVector<GradientPairInt64>;
template class HostDeviceVector<std::int32_t>;  // bst_node_t
template class HostDeviceVector<std::uint8_t>;
template class HostDeviceVector<std::int8_t>;
template class HostDeviceVector<FeatureType>;
template class HostDeviceVector<Entry>;
template class HostDeviceVector<bst_idx_t>;
template class HostDeviceVector<std::uint32_t>;  // bst_feature_t
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
