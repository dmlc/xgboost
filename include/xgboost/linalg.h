/*!
 * Copyright 2021 by XGBoost Contributors
 * \file linalg.h
 * \brief Linear algebra related utilities.
 */
#ifndef XGBOOST_LINALG_H_
#define XGBOOST_LINALG_H_

#include <xgboost/base.h>
#include <xgboost/generic_parameters.h>
#include <xgboost/span.h>

#include <algorithm>
#include <cassert>
#include <type_traits>
#include <utility>
#include <vector>

namespace xgboost {
namespace linalg {
namespace detail {
template <typename S, typename Head, size_t D>
constexpr size_t Offset(S (&strides)[D], size_t n, size_t dim, Head head) {
  assert(dim < D);
  return n + head * strides[dim];
}

template <typename S, size_t D, typename Head, typename... Tail>
constexpr size_t Offset(S (&strides)[D], size_t n, size_t dim, Head head, Tail &&...rest) {
  assert(dim < D);
  return Offset(strides, n + (head * strides[dim]), dim + 1, rest...);
}

struct AllTag {};
struct IntTag {};

/**
 * \brief Calculate the dimension of sliced tensor.
 */
template <typename T>
constexpr int32_t CalcSliceDim() {
  return std::is_same<T, IntTag>::value ? 0 : 1;
}

template <typename T, typename... S>
constexpr std::enable_if_t<sizeof...(S) != 0, int32_t> CalcSliceDim() {
  return CalcSliceDim<T>() + CalcSliceDim<S...>();
}

template <int32_t D>
constexpr size_t CalcSize(size_t (&shape)[D]) {
  size_t size = 1;
  for (auto d : shape) {
    size *= d;
  }
  return size;
}

template <typename S>
using RemoveCRType = std::remove_const_t<std::remove_reference_t<S>>;

template <typename S>
using IndexToTag = std::conditional_t<std::is_integral<RemoveCRType<S>>::value, IntTag, AllTag>;

template <int32_t n, typename Fn>
XGBOOST_DEVICE constexpr auto UnrollLoop(Fn fn) {
#if defined __CUDA_ARCH__
#pragma unroll n
#endif  // defined __CUDA_ARCH__
  for (int32_t i = 0; i < n; ++i) {
    fn(i);
  }
}
}  // namespace detail

/**
 * \brief Specify all elements in the axis is used for slice.
 */
constexpr detail::AllTag All() { return {}; }

/**
 * \brief A tensor view with static type and shape. It implements indexing and slicing.
 *
 * Most of the algorithms in XGBoost are implemented for both CPU and GPU without using
 * much linear algebra routines, this class is a helper intended to ease some high level
 * operations like indexing into prediction tensor or gradient matrix.  It can be passed
 * into CUDA kernel as normal argument for GPU algorithms.
 */
template <typename T, int32_t kDim = 5>
class TensorView {
 public:
  using ShapeT = size_t[kDim];
  using StrideT = ShapeT;

 private:
  StrideT stride_{1};
  ShapeT shape_{0};
  common::Span<T> data_;
  T* ptr_{nullptr};  // pointer of data_ to avoid bound check.

  size_t size_{0};
  int32_t device_{-1};

  // Unlike `Tensor`, the data_ can have arbitrary size since this is just a view.
  XGBOOST_DEVICE void CalcSize() {
    if (data_.empty()) {
      size_ = 0;
    } else {
      size_ = detail::CalcSize(shape_);
    }
  }

  struct SliceHelper {
    size_t old_dim;
    size_t new_dim;
    size_t offset;
  };

  template <int32_t D, typename... S>
  XGBOOST_DEVICE SliceHelper MakeSliceDim(size_t old_dim, size_t new_dim, size_t new_shape[D],
                                          size_t new_stride[D], detail::AllTag) const {
    new_stride[new_dim] = stride_[old_dim];
    new_shape[new_dim] = shape_[old_dim];
    return {old_dim + 1, new_dim + 1, 0};
  }

  template <int32_t D, typename... S>
  XGBOOST_DEVICE SliceHelper MakeSliceDim(size_t old_dim, size_t new_dim, size_t new_shape[D],
                                          size_t new_stride[D], detail::AllTag,
                                          S &&...slices) const {
    new_stride[new_dim] = stride_[old_dim];
    new_shape[new_dim] = shape_[old_dim];
    return MakeSliceDim<D>(old_dim + 1, new_dim + 1, new_shape, new_stride, slices...);
  }

  template <int32_t D, typename Index>
  XGBOOST_DEVICE SliceHelper MakeSliceDim(size_t old_dim, size_t new_dim, size_t new_shape[D],
                                          size_t new_stride[D], Index i) const {
    return {old_dim + 1, new_dim, stride_[old_dim] * i};
  }

  template <int32_t D, typename Index, typename... S>
  XGBOOST_DEVICE std::enable_if_t<std::is_integral<Index>::value, SliceHelper> MakeSliceDim(
      size_t old_dim, size_t new_dim, size_t new_shape[D], size_t new_stride[D], Index i,
      S &&...slices) const {
    auto offset = stride_[old_dim] * i;
    auto res = MakeSliceDim<D>(old_dim + 1, new_dim, new_shape, new_stride, slices...);
    return {res.old_dim, res.new_dim, res.offset + offset};
  }

 public:
  size_t constexpr static kValueSize = sizeof(T);
  size_t constexpr static kDimension = kDim;

 public:
  /**
   * \brief Create a tensor with data and shape.
   *
   * \tparam I     Type of the shape array element.
   * \tparam D     Size of the shape array, can be lesser than or equal to tensor dimension.
   *
   * \param data   Raw data input, can be const if this tensor has const type in its
   *               template parameter.
   * \param shape  shape of the tensor
   * \param device Device ordinal
   */
  template <typename I, int32_t D>
  XGBOOST_DEVICE TensorView(common::Span<T> data, I const (&shape)[D], int32_t device)
      : data_{data}, ptr_{data_.data()}, device_{device} {
    static_assert(D > 0 && D <= kDim, "Invalid shape.");
    // shape
    detail::UnrollLoop<D>([&](auto i) { shape_[i] = shape[i]; });
    for (auto i = D; i < kDim; ++i) {
      shape_[i] = 1;
    }
    // stride
    stride_[kDim - 1] = 1;
    for (int32_t s = kDim - 2; s >= 0; --s) {
      stride_[s] = shape_[s + 1] * stride_[s + 1];
    }
    this->CalcSize();
  };

  /**
   * \brief Create a tensor with data, shape and strides.  Don't use this constructor if
   *        stride can be calculated from shape.
   */
  template <typename I, int32_t D>
  XGBOOST_DEVICE TensorView(common::Span<T> data, I const (&shape)[D], I const (&stride)[D],
                            int32_t device)
      : data_{data}, ptr_{data_.data()}, device_{device} {
    static_assert(D == kDim, "Invalid shape & stride.");
    detail::UnrollLoop<D>([&](auto i) {
      shape_[i] = shape[i];
      stride_[i] = stride[i];
    });
    this->CalcSize();
  };

  XGBOOST_DEVICE TensorView(TensorView const &that)
      : data_{that.data_}, ptr_{data_.data()}, size_{that.size_}, device_{that.device_} {
    detail::UnrollLoop<kDim>([&](auto i) {
      stride_[i] = that.stride_[i];
      shape_[i] = that.shape_[i];
    });
  }

  /**
   * \brief Index the tensor to obtain a scalar value.
   *
   * \code
   *
   *   // Create a 3-dim tensor.
   *   Tensor<float, 3> t {data, shape, 0};
   *   float pi = 3.14159;
   *   t(1, 2, 3) = pi;
   *   ASSERT_EQ(t(1, 2, 3), pi);
   *
   * \endcode
   */
  template <typename... Index>
  XGBOOST_DEVICE T &operator()(Index &&...index) {
    static_assert(sizeof...(index) <= kDim, "Invalid index.");
    size_t offset = detail::Offset(stride_, 0ul, 0ul, index...);
    return ptr_[offset];
  }
  /**
   * \brief Index the tensor to obtain a scalar value.
   */
  template <typename... Index>
  XGBOOST_DEVICE T const &operator()(Index &&...index) const {
    static_assert(sizeof...(index) <= kDim, "Invalid index.");
    size_t offset = detail::Offset(stride_, 0ul, 0ul, index...);
    return ptr_[offset];
  }

  /**
   * \brief Slice the tensor.  The returned tensor has inferred dim and shape.
   *
   * \code
   *
   *   // Create a 3-dim tensor.
   *   Tensor<float, 3> t {data, shape, 0};
   *   // s has 2 dimensions (matrix)
   *   auto s = t.Slice(1, All(), All());
   *
   * \endcode
   */
  template <typename... S>
  XGBOOST_DEVICE auto Slice(S &&...slices) const {
    static_assert(sizeof...(slices) <= kDim, "Invalid slice.");
    int32_t constexpr kNewDim{detail::CalcSliceDim<detail::IndexToTag<S>...>()};
    size_t new_shape[kNewDim];
    size_t new_stride[kNewDim];
    auto res = MakeSliceDim<kNewDim>(size_t(0), size_t(0), new_shape, new_stride, slices...);
    // ret is a different type due to changed dimension, so we can not access its private
    // fields.
    TensorView<T, kNewDim> ret{data_.subspan(data_.empty() ? 0 : res.offset), new_shape, new_stride,
                               device_};
    return ret;
  }

  XGBOOST_DEVICE auto Shape() const { return common::Span<size_t const, kDim>{shape_}; }
  /**
   * Get the shape for i^th dimension
   */
  XGBOOST_DEVICE auto Shape(size_t i) const { return shape_[i]; }
  XGBOOST_DEVICE auto Stride() const { return common::Span<size_t const, kDim>{stride_}; }
  /**
   * Get the stride for i^th dimension, stride is specified as number of items instead of bytes.
   */
  XGBOOST_DEVICE auto Stride(size_t i) const { return stride_[i]; }

  XGBOOST_DEVICE auto cbegin() const { return data_.cbegin(); }  // NOLINT
  XGBOOST_DEVICE auto cend() const { return data_.cend(); }      // NOLINT
  XGBOOST_DEVICE auto begin() { return data_.begin(); }          // NOLINT
  XGBOOST_DEVICE auto end() { return data_.end(); }              // NOLINT

  XGBOOST_DEVICE size_t Size() const { return size_; }
  XGBOOST_DEVICE auto Values() const { return data_; }
  XGBOOST_DEVICE auto DeviceIdx() const { return device_; }
};

/**
 * \brief A view over a vector, specialization of Tensor
 *
 * \tparam T data type of vector
 */
template <typename T>
using VectorView = TensorView<T, 1>;

/**
 * \brief A view over a matrix, specialization of Tensor.
 *
 * \tparam T data type of matrix
 */
template <typename T>
using MatrixView = TensorView<T, 2>;
}  // namespace linalg
}  // namespace xgboost
#endif  // XGBOOST_LINALG_H_
