/*!
 * Copyright 2021 by Contributors
 * \file linalg.h
 * \brief  Linear algebra related utilities.
 */
#ifndef XGBOOST_LINALG_H_
#define XGBOOST_LINALG_H_

#include <xgboost/span.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/generic_parameters.h>

#include <array>
#include <algorithm>

namespace xgboost {
template <typename T> class MatrixView {
  int32_t device_;
  common::Span<T> values_;
  size_t strides_[2];
  size_t shape_[2];

 public:
  MatrixView(HostDeviceVector<T> *vec, std::array<size_t, 2> strides,
             std::array<size_t, 2> shape, int32_t device)
      : device_{device}, values_{device == GenericParameter::kCpuId
                                     ? vec->HostSpan()
                                     : vec->DeviceSpan()} {
    std::copy(strides.cbegin(), strides.cend(), strides_);
    std::copy(shape.cbegin(), shape.cend(), shape_);
  }
  MatrixView(HostDeviceVector<std::remove_const_t<T>> const *vec,
             std::array<size_t, 2> strides, std::array<size_t, 2> shape,
             int32_t device)
      : device_{device}, values_{device == GenericParameter::kCpuId
                                     ? vec->HostSpan()
                                     : vec->DeviceSpan()} {
    std::copy(strides.cbegin(), strides.cend(), strides_);
    std::copy(shape.cbegin(), shape.cend(), shape_);
  }
  XGBOOST_DEVICE T const &operator()(size_t r, size_t c) const {
    return values_[strides_[0] * r + strides_[1] * c];
  }

  auto Strides() const { return strides_; }
  auto Shape() const { return shape_; }
  auto Values() const { return values_; }
  auto Size() const { return shape_[0] * shape_[1]; }
  auto DeviceIdx() const { return device_; }
};

template <typename T, bool is_column = true> class VectorView {
  size_t strides_[2];
  size_t shape_[2];
  common::Span<T> values_;
  int32_t device_;
  size_t column_;
  static_assert(is_column, "Only column view over row matrix is implemented.");

 public:
  explicit VectorView(MatrixView<T> matrix, size_t column) {
    std::memcpy(strides_, matrix.Strides(), sizeof(strides_));
    std::memcpy(shape_, matrix.Shape(), sizeof(shape_));
    values_ = matrix.Values();
    column_ = column;
    device_ = matrix.DeviceIdx();
  }

  XGBOOST_DEVICE T &operator[](size_t i) {
    return values_[strides_[0] * i + strides_[1] * column_];
  }

  XGBOOST_DEVICE T const &operator[](size_t i) const {
    return values_[strides_[0] * i + strides_[1] * column_];
  }

  size_t Size() { return is_column ? shape_[0] : shape_[1]; }
  int32_t DeviceIdx() const { return device_; }
};
} // namespace xgboost
#endif  // XGBOOST_LINALG_H_
