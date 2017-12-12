/*!
 * Copyright 2017 XGBoost contributors
 */
#include "./dhvec.h"
#include "./device_helpers.cuh"

namespace xgboost {

template <typename T>
struct dhvec_impl {
  dhvec_impl(size_t size, int device)
    : device_(device), on_d_(device >= 0) {
    if (on_d_) {
      dh::safe_cuda(cudaSetDevice(device_));
      data_d_.resize(size);
    } else {
      data_h_.resize(size);
    }
  }
  dhvec_impl(const dhvec_impl<T>&) = delete;
  dhvec_impl(dhvec_impl<T>&&) = delete;
  void operator=(const dhvec_impl<T>&) = delete;
  void operator=(dhvec_impl<T>&&) = delete;

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
  void resize(size_t new_size, int new_device) {
    if (new_size == this->size() && new_device == device_)
      return;
    device_ = new_device;
    // if !on_d_, but the data size is 0 and the device is set,
    // resize the data on device instead
    if (!on_d_ && (data_h_.size() > 0 || device_ == -1)) {
      data_h_.resize(new_size);
    } else {
      dh::safe_cuda(cudaSetDevice(device_));
      data_d_.resize(new_size);
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
dhvec<T>::dhvec(size_t size, int device) : impl_(nullptr) {
  impl_ = new dhvec_impl<T>(size, device);
}

template <typename T>
dhvec<T>::~dhvec() {
  dhvec_impl<T>* tmp = impl_;
  impl_ = nullptr;
  delete tmp;
}

template <typename T>
size_t dhvec<T>::size() const { return impl_->size(); }

template <typename T>
int dhvec<T>::device() const { return impl_->device(); }

template <typename T>
T* dhvec<T>::ptr_d(int device) { return impl_->ptr_d(device); }

template <typename T>
thrust::device_ptr<T> dhvec<T>::tbegin(int device) {
  return impl_->tbegin(device);
}

template <typename T>
thrust::device_ptr<T> dhvec<T>::tend(int device) {
  return impl_->tend(device);
}

template <typename T>
std::vector<T>& dhvec<T>::data_h() { return impl_->data_h(); }

template <typename T>
void dhvec<T>::resize(size_t new_size, int new_device) {
  impl_->resize(new_size, new_device);
}

// explicit instantiations are required, as dhvec isn't header-only
template class dhvec<bst_float>;
template class dhvec<bst_gpair>;

}  // namespace xgboost
