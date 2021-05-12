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
#include <utility>

namespace xgboost {
/*!
 * \brief A view over a matrix on contiguous storage.
 *
 * \tparam T data type of matrix
 */
template <typename T> class MatrixView {
  int32_t device_;
  common::Span<T> values_;
  size_t strides_[2];
  size_t shape_[2];

  template <typename Vec> static auto InferValues(Vec *vec, int32_t device) {
    return device == GenericParameter::kCpuId ? vec->HostSpan()
                                              : vec->DeviceSpan();
  }

 public:
  /*!
   * \param vec     storage.
   * \param strides Strides for matrix.
   * \param shape   Rows and columns.
   * \param device  Where the data is stored in.
   */
  MatrixView(HostDeviceVector<T> *vec, std::array<size_t, 2> strides,
             std::array<size_t, 2> shape, int32_t device)
      : device_{device}, values_{InferValues(vec, device)} {
    std::copy(strides.cbegin(), strides.cend(), strides_);
    std::copy(shape.cbegin(), shape.cend(), shape_);
  }
  MatrixView(HostDeviceVector<std::remove_const_t<T>> const *vec,
             std::array<size_t, 2> strides, std::array<size_t, 2> shape,
             int32_t device)
      : device_{device}, values_{InferValues(vec, device)} {
    std::copy(strides.cbegin(), strides.cend(), strides_);
    std::copy(shape.cbegin(), shape.cend(), shape_);
  }
  /*! \brief Row major constructor. */
  MatrixView(HostDeviceVector<T> *vec, std::array<size_t, 2> shape,
             int32_t device)
      : device_{device}, values_{InferValues(vec, device)} {
    std::copy(shape.cbegin(), shape.cend(), shape_);
    strides_[0] = shape[1];
    strides_[1] = 1;
  }
  MatrixView(HostDeviceVector<std::remove_const_t<T>> const *vec,
             std::array<size_t, 2> shape, int32_t device)
      : device_{device}, values_{InferValues(vec, device)} {
    std::copy(shape.cbegin(), shape.cend(), shape_);
    strides_[0] = shape[1];
    strides_[1] = 1;
  }

  XGBOOST_DEVICE T const &operator()(size_t r, size_t c) const {
    return values_[strides_[0] * r + strides_[1] * c];
  }
  XGBOOST_DEVICE T &operator()(size_t r, size_t c) {
    return values_[strides_[0] * r + strides_[1] * c];
  }

  auto Strides() const { return strides_; }
  auto Shape() const { return shape_; }
  auto Values() const { return values_; }
  auto Size() const { return shape_[0] * shape_[1]; }
  auto DeviceIdx() const { return device_; }
};

/*! \brief A slice for 1 column of MatrixView.  Can be extended to row if needed. */
template <typename T> class VectorView {
  MatrixView<T> matrix_;
  size_t column_;

 public:
  explicit VectorView(MatrixView<T> matrix, size_t column)
      : matrix_{std::move(matrix)}, column_{column} {}

  XGBOOST_DEVICE T &operator[](size_t i) {
    return matrix_(i, column_);
  }

  XGBOOST_DEVICE T const &operator[](size_t i) const {
    return matrix_(i, column_);
  }

  size_t Size() { return matrix_.Shape()[0]; }
  int32_t DeviceIdx() const { return matrix_.DeviceIdx(); }
};
}       // namespace xgboost
#endif  // XGBOOST_LINALG_H_
