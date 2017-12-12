/*!
 * Copyright 2017 XGBoost contributors
 */
#ifndef XGBOOST_USE_CUDA

// dummy implementation of dhvec in case CUDA is not used

#include <xgboost/base.h>
#include "./dhvec.h"

namespace xgboost {

template <typename T>
struct dhvec_impl {
  explicit dhvec_impl(size_t size) : data_h_(size) {}
  std::vector<T> data_h_;
};

template <typename T>
dhvec<T>::dhvec(size_t size, int device) : impl_(nullptr) {
  impl_ = new dhvec_impl<T>(size);
}

template <typename T>
dhvec<T>::~dhvec() {
  dhvec_impl<T>* tmp = impl_;
  impl_ = nullptr;
  delete tmp;
}

template <typename T>
size_t dhvec<T>::size() const { return impl_->data_h_.size(); }

template <typename T>
int dhvec<T>::device() const { return -1; }

template <typename T>
T* dhvec<T>::ptr_d(int device) { return nullptr; }

template <typename T>
std::vector<T>& dhvec<T>::data_h() { return impl_->data_h_; }

template <typename T>
void dhvec<T>::resize(size_t new_size, int new_device) {
  impl_->data_h_.resize(new_size);
}

// explicit instantiations are required, as dhvec isn't header-only
template class dhvec<bst_float>;
template class dhvec<bst_gpair>;

}  // namespace xgboost

#endif
