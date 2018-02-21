/*!
 * Copyright 2017 XGBoost contributors
 */

#include "./host_device_vector.h"
#include "./device_helpers.cuh"

namespace xgboost {

template <typename T>
struct HostDeviceVectorImpl {
  HostDeviceVectorImpl(size_t size, int device, T v)
    : device_(device), on_d_(device >= 0) {
    if (on_d_) {
      dh::safe_cuda(cudaSetDevice(device_));
      data_d_.resize(size, v);
    } else {
      data_h_.resize(size, v);
    }
  }
  // Init can be std::vector<T> or std::initializer_list<T>
  template <class Init>
  HostDeviceVectorImpl(const Init& init, int device)
    : device_(device), on_d_(device >= 0) {
    if (on_d_) {
      dh::safe_cuda(cudaSetDevice(device_));
      data_d_.resize(init.size());
      thrust::copy(init.begin(), init.end(), data_d_.begin());
    } else {
      data_h_ = init;
    }
  }
  HostDeviceVectorImpl(const HostDeviceVectorImpl<T>&) = delete;
  HostDeviceVectorImpl(HostDeviceVectorImpl<T>&&) = delete;
  void operator=(const HostDeviceVectorImpl<T>&) = delete;
  void operator=(HostDeviceVectorImpl<T>&&) = delete;

  size_t size() const { return on_d_ ? data_d_.size() : data_h_.size(); }

  int device() const { return device_; }

  T* ptr_d(int device) {
    lazy_sync_device(device);
    return data_d_.data().get();
  }
  thrust::device_ptr<T> tbegin(int device) {
    return thrust::device_ptr<T>(ptr_d(device));
  }
  thrust::device_ptr<T> tend(int device) {
    auto begin = tbegin(device);
    return begin + size();
  }
  std::vector<T>& data_h() {
    lazy_sync_host();
    return data_h_;
  }
  void resize(size_t new_size, int new_device, T v) {
    if (new_size == this->size() && new_device == device_)
      return;
    if (new_device != -1)
      device_ = new_device;
    // if !on_d_, but the data size is 0 and the device is set,
    // resize the data on device instead
    if (!on_d_ && (data_h_.size() > 0 || device_ == -1)) {
      data_h_.resize(new_size, v);
    } else {
      dh::safe_cuda(cudaSetDevice(device_));
      data_d_.resize(new_size, v);
      on_d_ = true;
    }
  }

  void lazy_sync_host() {
    if (!on_d_)
      return;
    if (data_h_.size() != this->size())
      data_h_.resize(this->size());
    dh::safe_cuda(cudaSetDevice(device_));
    thrust::copy(data_d_.begin(), data_d_.end(), data_h_.begin());
    on_d_ = false;
  }

  void lazy_sync_device(int device) {
    if (on_d_)
      return;
    if (device != device_) {
      CHECK_EQ(device_, -1);
      device_ = device;
    }
    if (data_d_.size() != this->size()) {
      dh::safe_cuda(cudaSetDevice(device_));
      data_d_.resize(this->size());
    }
    dh::safe_cuda(cudaSetDevice(device_));
    thrust::copy(data_h_.begin(), data_h_.end(), data_d_.begin());
    on_d_ = true;
  }

  std::vector<T> data_h_;
  thrust::device_vector<T> data_d_;
  // true if there is an up-to-date copy of data on device, false otherwise
  bool on_d_;
  int device_;
};

template <typename T>
HostDeviceVector<T>::HostDeviceVector(size_t size, int device, T v) : impl_(nullptr) {
  impl_ = new HostDeviceVectorImpl<T>(size, device, v);
}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(std::initializer_list<T> init, int device)
  : impl_(nullptr) {
  impl_ = new HostDeviceVectorImpl<T>(init, device);
}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(const std::vector<T>& init, int device)
  : impl_(nullptr) {
  impl_ = new HostDeviceVectorImpl<T>(init, device);
}

template <typename T>
HostDeviceVector<T>::~HostDeviceVector() {
  HostDeviceVectorImpl<T>* tmp = impl_;
  impl_ = nullptr;
  delete tmp;
}

template <typename T>
size_t HostDeviceVector<T>::size() const { return impl_->size(); }

template <typename T>
int HostDeviceVector<T>::device() const { return impl_->device(); }

template <typename T>
T* HostDeviceVector<T>::ptr_d(int device) { return impl_->ptr_d(device); }

template <typename T>
thrust::device_ptr<T> HostDeviceVector<T>::tbegin(int device) {
  return impl_->tbegin(device);
}

template <typename T>
thrust::device_ptr<T> HostDeviceVector<T>::tend(int device) {
  return impl_->tend(device);
}

template <typename T>
std::vector<T>& HostDeviceVector<T>::data_h() { return impl_->data_h(); }

template <typename T>
void HostDeviceVector<T>::resize(size_t new_size, int new_device, T v) {
  impl_->resize(new_size, new_device, v);
}

// explicit instantiations are required, as HostDeviceVector isn't header-only
template class HostDeviceVector<bst_float>;
template class HostDeviceVector<bst_gpair>;

}  // namespace xgboost
