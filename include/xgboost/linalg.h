/**
 * Copyright 2021-2023 by XGBoost Contributors
 * \file linalg.h
 * \brief Linear algebra related utilities.
 */
#ifndef XGBOOST_LINALG_H_
#define XGBOOST_LINALG_H_

#include <dmlc/endian.h>
#include <xgboost/base.h>
#include <xgboost/context.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/json.h>
#include <xgboost/span.h>

#include <algorithm>
#include <cassert>
#include <cinttypes>  // for int32_t
#include <cstddef>    // for size_t
#include <limits>
#include <string>
#include <tuple>  // for make_tuple
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif  // defined(_MSC_VER)

// decouple it from xgboost.
#ifndef LINALG_HD
#if defined(__CUDA__) || defined(__NVCC__)
#define LINALG_HD __host__ __device__
#else
#define LINALG_HD
#endif  // defined (__CUDA__) || defined(__NVCC__)
#endif  // LINALG_HD

namespace xgboost::linalg {
namespace detail {

struct ArrayInterfaceHandler {
  template <typename T>
  static constexpr char TypeChar() {
    return (std::is_floating_point<T>::value
                ? 'f'
                : (std::is_integral<T>::value ? (std::is_signed<T>::value ? 'i' : 'u') : '\0'));
  }
};

template <size_t dim, typename S, typename Head, size_t D>
constexpr size_t Offset(S (&strides)[D], size_t n, Head head) {
  static_assert(dim < D);
  return n + head * strides[dim];
}

template <size_t dim, typename S, size_t D, typename Head, typename... Tail>
constexpr std::enable_if_t<sizeof...(Tail) != 0, size_t> Offset(S (&strides)[D], size_t n,
                                                                Head head, Tail &&...rest) {
  static_assert(dim < D);
  return Offset<dim + 1>(strides, n + (head * strides[dim]), std::forward<Tail>(rest)...);
}

template <int32_t D, bool f_array = false>
constexpr void CalcStride(size_t const (&shape)[D], size_t (&stride)[D]) {
  if (f_array) {
    stride[0] = 1;
    for (int32_t s = 1; s < D; ++s) {
      stride[s] = shape[s - 1] * stride[s - 1];
    }
  } else {
    stride[D - 1] = 1;
    for (int32_t s = D - 2; s >= 0; --s) {
      stride[s] = shape[s + 1] * stride[s + 1];
    }
  }
}

struct AllTag {};

struct IntTag {};

template <typename I>
struct RangeTag {
  I beg;
  I end;
  [[nodiscard]] constexpr size_t Size() const { return end - beg; }
};

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
using IndexToTag = std::conditional_t<std::is_integral<RemoveCRType<S>>::value, IntTag, S>;

template <int32_t n, typename Fn>
LINALG_HD constexpr auto UnrollLoop(Fn fn) {
#if defined __CUDA_ARCH__
#pragma unroll n
#endif  // defined __CUDA_ARCH__
  for (int32_t i = 0; i < n; ++i) {
    fn(i);
  }
}

template <typename T>
int32_t NativePopc(T v) {
  int c = 0;
  for (; v != 0; v &= v - 1) c++;
  return c;
}

inline LINALG_HD int Popc(uint32_t v) {
#if defined(__CUDA_ARCH__)
  return __popc(v);
#elif defined(__GNUC__) || defined(__clang__)
  return __builtin_popcount(v);
#elif defined(_MSC_VER)
  return __popcnt(v);
#else
  return NativePopc(v);
#endif  // compiler
}

inline LINALG_HD int Popc(uint64_t v) {
#if defined(__CUDA_ARCH__)
  return __popcll(v);
#elif defined(__GNUC__) || defined(__clang__)
  return __builtin_popcountll(v);
#elif defined(_MSC_VER) && defined(_M_X64)
  return __popcnt64(v);
#else
  return NativePopc(v);
#endif  // compiler
}

template <std::size_t D, typename Head>
LINALG_HD void IndexToArr(std::size_t (&arr)[D], Head head) {
  static_assert(std::is_integral<std::remove_reference_t<Head>>::value, "Invalid index type.");
  arr[D - 1] = head;
}

/**
 * \brief Convert index from parameter pack to C-style array.
 */
template <std::size_t D, typename Head, typename... Rest>
LINALG_HD void IndexToArr(std::size_t (&arr)[D], Head head, Rest &&...index) {
  static_assert(sizeof...(Rest) < D, "Index overflow.");
  static_assert(std::is_integral<std::remove_reference_t<Head>>::value, "Invalid index type.");
  arr[D - sizeof...(Rest) - 1] = head;
  IndexToArr(arr, std::forward<Rest>(index)...);
}

template <class T, std::size_t N, std::size_t... Idx>
constexpr auto ArrToTuple(T (&arr)[N], std::index_sequence<Idx...>) {
  return std::make_tuple(arr[Idx]...);
}

/**
 * \brief Convert C-styple array to std::tuple.
 */
template <class T, std::size_t N>
constexpr auto ArrToTuple(T (&arr)[N]) {
  return ArrToTuple(arr, std::make_index_sequence<N>{});
}

// uint division optimization inspired by the CIndexer in cupy.  Division operation is
// slow on both CPU and GPU, especially 64 bit integer.  So here we first try to avoid 64
// bit when the index is smaller, then try to avoid division when it's exp of 2.
template <typename I, int32_t D>
LINALG_HD auto UnravelImpl(I idx, common::Span<size_t const, D> shape) {
  size_t index[D]{0};
  static_assert(std::is_signed<decltype(D)>::value,
                "Don't change the type without changing the for loop.");
  for (int32_t dim = D; --dim > 0;) {
    auto s = static_cast<std::remove_const_t<std::remove_reference_t<I>>>(shape[dim]);
    if (s & (s - 1)) {
      auto t = idx / s;
      index[dim] = idx - t * s;
      idx = t;
    } else {  // exp of 2
      index[dim] = idx & (s - 1);
      idx >>= Popc(s - 1);
    }
  }
  index[0] = idx;
  return ArrToTuple(index);
}

template <size_t dim, typename I, int32_t D>
void ReshapeImpl(size_t (&out_shape)[D], I s) {
  static_assert(dim < D);
  out_shape[dim] = s;
}

template <size_t dim, int32_t D, typename... S, typename I,
          std::enable_if_t<sizeof...(S) != 0> * = nullptr>
void ReshapeImpl(size_t (&out_shape)[D], I &&s, S &&...rest) {
  static_assert(dim < D);
  out_shape[dim] = s;
  ReshapeImpl<dim + 1>(out_shape, std::forward<S>(rest)...);
}

template <typename Fn, typename Tup, size_t... I>
LINALG_HD decltype(auto) constexpr Apply(Fn &&f, Tup &&t, std::index_sequence<I...>) {
  return f(std::get<I>(t)...);
}

/**
 * C++ 17 style apply.
 *
 * \param f function to apply
 * \param t tuple of arguments
 */
template <typename Fn, typename Tup>
LINALG_HD decltype(auto) constexpr Apply(Fn &&f, Tup &&t) {
  constexpr auto kSize = std::tuple_size<Tup>::value;
  return Apply(std::forward<Fn>(f), std::forward<Tup>(t), std::make_index_sequence<kSize>{});
}

/**
 * C++ 17 conjunction
 */
template <class...>
struct Conjunction : std::true_type {};
template <class B1>
struct Conjunction<B1> : B1 {};
template <class B1, class... Bn>
struct Conjunction<B1, Bn...>
    : std::conditional_t<static_cast<bool>(B1::value), Conjunction<Bn...>, B1> {};

template <typename... Index>
using IsAllIntegral = Conjunction<std::is_integral<std::remove_reference_t<Index>>...>;

template <typename... Index>
using EnableIfIntegral = std::enable_if_t<IsAllIntegral<Index...>::value>;
}  // namespace detail

/**
 * \brief Specify all elements in the axis for slicing.
 */
constexpr detail::AllTag All() { return {}; }
/**
 * \brief Specify a range of elements in the axis for slicing.
 */
template <typename I>
constexpr detail::RangeTag<I> Range(I beg, I end) {
  return {beg, end};
}

enum Order : std::uint8_t {
  kC,  // Row major
  kF,  // Col major
};

/**
 * \brief A tensor view with static type and dimension. It implements indexing and slicing.
 *
 * Most of the algorithms in XGBoost are implemented for both CPU and GPU without using
 * much linear algebra routines, this class is a helper intended to ease some high level
 * operations like indexing into prediction tensor or gradient matrix.  It can be passed
 * into CUDA kernel as normal argument for GPU algorithms.
 *
 * Ideally we should add a template parameter `bool on_host` so that the compiler can
 * prevent passing/accessing the wrong view, but inheritance is heavily used in XGBoost so
 * some functions expect data types that can be used in everywhere (update prediction
 * cache for example).
 */
template <typename T, int32_t kDim>
class TensorView {
 public:
  using ShapeT = size_t[kDim];
  using StrideT = ShapeT;

 private:
  StrideT stride_{1};
  ShapeT shape_{0};
  common::Span<T> data_;
  T *ptr_{nullptr};  // pointer of data_ to avoid bound check.

  size_t size_{0};
  int32_t device_{-1};

  // Unlike `Tensor`, the data_ can have arbitrary size since this is just a view.
  LINALG_HD void CalcSize() {
    if (data_.empty()) {
      size_ = 0;
    } else {
      size_ = detail::CalcSize(shape_);
    }
  }

  template <size_t old_dim, size_t new_dim, int32_t D, typename I>
  LINALG_HD size_t MakeSliceDim(size_t new_shape[D], size_t new_stride[D],
                                detail::RangeTag<I> &&range) const {
    static_assert(new_dim < D);
    static_assert(old_dim < kDim);
    new_stride[new_dim] = stride_[old_dim];
    new_shape[new_dim] = range.Size();
    assert(static_cast<decltype(shape_[old_dim])>(range.end) <= shape_[old_dim]);

    auto offset = stride_[old_dim] * range.beg;
    return offset;
  }
  /**
   * \brief Slice dimension for Range tag.
   */
  template <size_t old_dim, size_t new_dim, int32_t D, typename I, typename... S>
  LINALG_HD size_t MakeSliceDim(size_t new_shape[D], size_t new_stride[D],
                                detail::RangeTag<I> &&range, S &&...slices) const {
    static_assert(new_dim < D);
    static_assert(old_dim < kDim);
    new_stride[new_dim] = stride_[old_dim];
    new_shape[new_dim] = range.Size();
    assert(static_cast<decltype(shape_[old_dim])>(range.end) <= shape_[old_dim]);

    auto offset = stride_[old_dim] * range.beg;
    return MakeSliceDim<old_dim + 1, new_dim + 1, D>(new_shape, new_stride,
                                                     std::forward<S>(slices)...) +
           offset;
  }

  template <size_t old_dim, size_t new_dim, int32_t D>
  LINALG_HD size_t MakeSliceDim(size_t new_shape[D], size_t new_stride[D], detail::AllTag) const {
    static_assert(new_dim < D);
    static_assert(old_dim < kDim);
    new_stride[new_dim] = stride_[old_dim];
    new_shape[new_dim] = shape_[old_dim];
    return 0;
  }
  /**
   * \brief Slice dimension for All tag.
   */
  template <size_t old_dim, size_t new_dim, int32_t D, typename... S>
  LINALG_HD size_t MakeSliceDim(size_t new_shape[D], size_t new_stride[D], detail::AllTag,
                                S &&...slices) const {
    static_assert(new_dim < D);
    static_assert(old_dim < kDim);
    new_stride[new_dim] = stride_[old_dim];
    new_shape[new_dim] = shape_[old_dim];
    return MakeSliceDim<old_dim + 1, new_dim + 1, D>(new_shape, new_stride,
                                                     std::forward<S>(slices)...);
  }

  template <size_t old_dim, size_t new_dim, int32_t D, typename Index>
  LINALG_HD size_t MakeSliceDim(DMLC_ATTRIBUTE_UNUSED size_t new_shape[D],
                                DMLC_ATTRIBUTE_UNUSED size_t new_stride[D], Index i) const {
    static_assert(old_dim < kDim);
    return stride_[old_dim] * i;
  }
  /**
   * \brief Slice dimension for Index tag.
   */
  template <size_t old_dim, size_t new_dim, int32_t D, typename Index, typename... S>
  LINALG_HD std::enable_if_t<std::is_integral<Index>::value, size_t> MakeSliceDim(
      size_t new_shape[D], size_t new_stride[D], Index i, S &&...slices) const {
    static_assert(old_dim < kDim);
    auto offset = stride_[old_dim] * i;
    auto res =
        MakeSliceDim<old_dim + 1, new_dim, D>(new_shape, new_stride, std::forward<S>(slices)...);
    return res + offset;
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
  LINALG_HD TensorView(common::Span<T> data, I const (&shape)[D], std::int32_t device)
      : TensorView{data, shape, device, Order::kC} {}

  template <typename I, int32_t D>
  LINALG_HD TensorView(common::Span<T> data, I const (&shape)[D], std::int32_t device, Order order)
      : data_{data}, ptr_{data_.data()}, device_{device} {
    static_assert(D > 0 && D <= kDim, "Invalid shape.");
    // shape
    detail::UnrollLoop<D>([&](auto i) { shape_[i] = shape[i]; });
    for (auto i = D; i < kDim; ++i) {
      shape_[i] = 1;
    }
    // stride
    switch (order) {
      case Order::kC: {
        detail::CalcStride(shape_, stride_);
        break;
      }
      case Order::kF: {
        detail::CalcStride<kDim, true>(shape_, stride_);
        break;
      }
      default: {
        SPAN_CHECK(false);
      }
    }
    // size
    this->CalcSize();
  }

  /**
   * \brief Create a tensor with data, shape and strides.  Don't use this constructor if
   *        stride can be calculated from shape.
   */
  template <typename I, std::int32_t D>
  LINALG_HD TensorView(common::Span<T> data, I const (&shape)[D], I const (&stride)[D],
                       std::int32_t device)
      : data_{data}, ptr_{data_.data()}, device_{device} {
    static_assert(D == kDim, "Invalid shape & stride.");
    detail::UnrollLoop<D>([&](auto i) {
      shape_[i] = shape[i];
      stride_[i] = stride[i];
    });
    this->CalcSize();
  }

  template <
      typename U,
      std::enable_if_t<common::detail::IsAllowedElementTypeConversion<U, T>::value> * = nullptr>
  LINALG_HD TensorView(TensorView<U, kDim> const &that)  // NOLINT
      : data_{that.Values()}, ptr_{data_.data()}, size_{that.Size()}, device_{that.DeviceIdx()} {
    detail::UnrollLoop<kDim>([&](auto i) {
      stride_[i] = that.Stride(i);
      shape_[i] = that.Shape(i);
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
  template <typename... Index, detail::EnableIfIntegral<Index...> * = nullptr>
  LINALG_HD T &operator()(Index &&...index) {
    static_assert(sizeof...(index) <= kDim, "Invalid index.");
    size_t offset = detail::Offset<0ul>(stride_, 0ul, std::forward<Index>(index)...);
    assert(offset < data_.size() && "Out of bound access.");
    return ptr_[offset];
  }
  /**
   * \brief Index the tensor to obtain a scalar value.
   */
  template <typename... Index, detail::EnableIfIntegral<Index...> * = nullptr>
  LINALG_HD T const &operator()(Index &&...index) const {
    static_assert(sizeof...(index) <= kDim, "Invalid index.");
    size_t offset = detail::Offset<0ul>(stride_, 0ul, std::forward<Index>(index)...);
    assert(offset < data_.size() && "Out of bound access.");
    return ptr_[offset];
  }

  /**
   * \brief Slice the tensor.  The returned tensor has inferred dim and shape.  Scalar
   *        result is not supported.
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
  LINALG_HD auto Slice(S &&...slices) const {
    static_assert(sizeof...(slices) <= kDim, "Invalid slice.");
    int32_t constexpr kNewDim{detail::CalcSliceDim<detail::IndexToTag<S>...>()};
    size_t new_shape[kNewDim];
    size_t new_stride[kNewDim];
    auto offset = MakeSliceDim<0, 0, kNewDim>(new_shape, new_stride, std::forward<S>(slices)...);
    // ret is a different type due to changed dimension, so we can not access its private
    // fields.
    TensorView<T, kNewDim> ret{data_.subspan(data_.empty() ? 0 : offset), new_shape, new_stride,
                               device_};
    return ret;
  }

  LINALG_HD auto Shape() const { return common::Span<size_t const, kDim>{shape_}; }
  /**
   * Get the shape for i^th dimension
   */
  LINALG_HD auto Shape(size_t i) const { return shape_[i]; }
  LINALG_HD auto Stride() const { return common::Span<size_t const, kDim>{stride_}; }
  /**
   * Get the stride for i^th dimension, stride is specified as number of items instead of bytes.
   */
  LINALG_HD auto Stride(size_t i) const { return stride_[i]; }

  /**
   * \brief Number of items in the tensor.
   */
  [[nodiscard]] LINALG_HD std::size_t Size() const { return size_; }
  /**
   * \brief Whether this is a contiguous array, both C and F contiguous returns true.
   */
  [[nodiscard]] LINALG_HD bool Contiguous() const {
    return data_.size() == this->Size() || this->CContiguous() || this->FContiguous();
  }
  /**
   * \brief Whether it's a c-contiguous array.
   */
  [[nodiscard]] LINALG_HD bool CContiguous() const {
    StrideT stride;
    static_assert(std::is_same<decltype(stride), decltype(stride_)>::value);
    // It's contiguous if the stride can be calculated from shape.
    detail::CalcStride(shape_, stride);
    return common::Span<size_t const, kDim>{stride_} == common::Span<size_t const, kDim>{stride};
  }
  /**
   * \brief Whether it's a f-contiguous array.
   */
  [[nodiscard]] LINALG_HD bool FContiguous() const {
    StrideT stride;
    static_assert(std::is_same<decltype(stride), decltype(stride_)>::value);
    // It's contiguous if the stride can be calculated from shape.
    detail::CalcStride<kDim, true>(shape_, stride);
    return common::Span<size_t const, kDim>{stride_} == common::Span<size_t const, kDim>{stride};
  }
  /**
   * \brief Obtain a reference to the raw data.
   */
  LINALG_HD auto Values() const -> decltype(data_) const & { return data_; }
  /**
   * \brief Obtain the CUDA device ordinal.
   */
  LINALG_HD auto DeviceIdx() const { return device_; }
};

/**
 * \brief Constructor for automatic type deduction.
 */
template <typename Container, typename... S,
          std::enable_if_t<!common::detail::IsSpan<Container>::value &&
                           !std::is_pointer_v<Container>> * = nullptr>
auto MakeTensorView(Context const *ctx, Container &data, S &&...shape) {  // NOLINT
  using T = typename Container::value_type;
  std::size_t in_shape[sizeof...(S)];
  detail::IndexToArr(in_shape, std::forward<S>(shape)...);
  return TensorView<T, sizeof...(S)>{data, in_shape, ctx->gpu_id};
}

template <typename T, typename... S>
LINALG_HD auto MakeTensorView(std::int32_t device, common::Span<T> data, S &&...shape) {
  std::size_t in_shape[sizeof...(S)];
  detail::IndexToArr(in_shape, std::forward<S>(shape)...);
  return TensorView<T, sizeof...(S)>{data, in_shape, device};
}

template <typename T, typename... S>
auto MakeTensorView(Context const *ctx, common::Span<T> data, S &&...shape) {
  return MakeTensorView(ctx->gpu_id, data, std::forward<S>(shape)...);
}

template <typename T, typename... S>
auto MakeTensorView(Context const *ctx, HostDeviceVector<T> *data, S &&...shape) {
  auto span = ctx->IsCPU() ? data->HostSpan() : data->DeviceSpan();
  return MakeTensorView(ctx->gpu_id, span, std::forward<S>(shape)...);
}

template <typename T, typename... S>
auto MakeTensorView(Context const *ctx, HostDeviceVector<T> const *data, S &&...shape) {
  auto span = ctx->IsCPU() ? data->ConstHostSpan() : data->ConstDeviceSpan();
  return MakeTensorView(ctx->gpu_id, span, std::forward<S>(shape)...);
}

/**
 * \brief Turns linear index into multi-dimension index.  Similar to numpy unravel.
 */
template <size_t D>
LINALG_HD auto UnravelIndex(size_t idx, common::Span<size_t const, D> shape) {
  if (idx > std::numeric_limits<uint32_t>::max()) {
    return detail::UnravelImpl<uint64_t, D>(static_cast<uint64_t>(idx), shape);
  } else {
    return detail::UnravelImpl<uint32_t, D>(static_cast<uint32_t>(idx), shape);
  }
}

template <size_t D>
LINALG_HD auto UnravelIndex(size_t idx, std::size_t const (&shape)[D]) {
  return UnravelIndex(idx, common::Span<std::size_t const, D>(shape));
}

template <typename... S>
LINALG_HD auto UnravelIndex(std::size_t idx, S... shape) {
  std::size_t s[sizeof...(S)];
  detail::IndexToArr(s, shape...);
  return UnravelIndex(idx, common::Span<std::size_t const, sizeof...(S)>(s));
}

/**
 * \brief A view over a vector, specialization of Tensor
 *
 * \tparam T data type of vector
 */
template <typename T>
using VectorView = TensorView<T, 1>;

/**
 * \brief Create a vector view from contigious memory.
 *
 * \param ptr Pointer to the contigious memory.
 * \param s   Size of the vector.
 * \param device (optional) Device ordinal, default to be host.
 */
template <typename T>
auto MakeVec(T *ptr, size_t s, int32_t device = -1) {
  return linalg::TensorView<T, 1>{{ptr, s}, {s}, device};
}

template <typename T>
auto MakeVec(HostDeviceVector<T> *data) {
  return MakeVec(data->DeviceIdx() == -1 ? data->HostPointer() : data->DevicePointer(),
                 data->Size(), data->DeviceIdx());
}

template <typename T>
auto MakeVec(HostDeviceVector<T> const *data) {
  return MakeVec(data->DeviceIdx() == -1 ? data->ConstHostPointer() : data->ConstDevicePointer(),
                 data->Size(), data->DeviceIdx());
}

/**
 * \brief A view over a matrix, specialization of Tensor.
 *
 * \tparam T data type of matrix
 */
template <typename T>
using MatrixView = TensorView<T, 2>;

/**
 * \brief Array Interface defined by
 * <a href="https://numpy.org/doc/stable/reference/arrays.interface.html">numpy</a>.
 *
 * `stream` is optionally included when data is on CUDA device.
 */
template <typename T, int32_t D>
Json ArrayInterface(TensorView<T const, D> const &t) {
  Json array_interface{Object{}};
  array_interface["data"] = std::vector<Json>(2);
  array_interface["data"][0] = Integer{reinterpret_cast<int64_t>(t.Values().data())};
  array_interface["data"][1] = Boolean{true};
  if (t.DeviceIdx() >= 0) {
    // Change this once we have different CUDA stream.
    array_interface["stream"] = Null{};
  }
  std::vector<Json> shape(t.Shape().size());
  std::vector<Json> stride(t.Stride().size());
  for (size_t i = 0; i < t.Shape().size(); ++i) {
    shape[i] = Integer(t.Shape(i));
    stride[i] = Integer(t.Stride(i) * sizeof(T));
  }
  array_interface["shape"] = Array{shape};
  array_interface["strides"] = Array{stride};
  array_interface["version"] = 3;

  char constexpr kT = detail::ArrayInterfaceHandler::TypeChar<T>();
  static_assert(kT != '\0');
  if (DMLC_LITTLE_ENDIAN) {
    array_interface["typestr"] = String{"<" + (kT + std::to_string(sizeof(T)))};
  } else {
    array_interface["typestr"] = String{">" + (kT + std::to_string(sizeof(T)))};
  }
  return array_interface;
}

/**
 * \brief Same as const version, but returns non-readonly data pointer.
 */
template <typename T, int32_t D>
Json ArrayInterface(TensorView<T, D> const &t) {
  TensorView<T const, D> const &as_const = t;
  auto res = ArrayInterface(as_const);
  res["data"][1] = Boolean{false};
  return res;
}

/**
 * \brief Return string representation of array interface.
 */
template <typename T, int32_t D>
auto ArrayInterfaceStr(TensorView<T const, D> const &t) {
  std::string str;
  Json::Dump(ArrayInterface(t), &str);
  return str;
}

template <typename T, int32_t D>
auto ArrayInterfaceStr(TensorView<T, D> const &t) {
  std::string str;
  Json::Dump(ArrayInterface(t), &str);
  return str;
}

/**
 * \brief A tensor storage. To use it for other functionality like slicing one needs to
 *        obtain a view first.  This way we can use it on both host and device.
 */
template <typename T, int32_t kDim = 5>
class Tensor {
 public:
  using ShapeT = size_t[kDim];
  using StrideT = ShapeT;

 private:
  HostDeviceVector<T> data_;
  ShapeT shape_{0};
  Order order_{Order::kC};

  template <typename I, std::int32_t D>
  void Initialize(I const (&shape)[D], std::int32_t device) {
    static_assert(D <= kDim, "Invalid shape.");
    std::copy(shape, shape + D, shape_);
    for (auto i = D; i < kDim; ++i) {
      shape_[i] = 1;
    }
    if (device >= 0) {
      data_.SetDevice(device);
      data_.ConstDevicePointer();  // Pull to device;
    }
    CHECK_EQ(data_.Size(), detail::CalcSize(shape_));
  }

 public:
  Tensor() = default;

  /**
   * \brief Create a tensor with shape and device ordinal.  The storage is initialized
   *        automatically.
   *
   * See \ref TensorView for parameters of this constructor.
   */
  template <typename I, int32_t D>
  explicit Tensor(I const (&shape)[D], std::int32_t device, Order order = kC)
      : Tensor{common::Span<I const, D>{shape}, device, order} {}

  template <typename I, size_t D>
  explicit Tensor(common::Span<I const, D> shape, std::int32_t device, Order order = kC)
      : order_{order} {
    // No device unroll as this is a host only function.
    std::copy(shape.data(), shape.data() + D, shape_);
    for (auto i = D; i < kDim; ++i) {
      shape_[i] = 1;
    }
    auto size = detail::CalcSize(shape_);
    if (device >= 0) {
      data_.SetDevice(device);
    }
    data_.Resize(size);
    if (device >= 0) {
      data_.DevicePointer();  // Pull to device
    }
  }
  /**
   * Initialize from 2 host iterators.
   */
  template <typename It, typename I, int32_t D>
  explicit Tensor(It begin, It end, I const (&shape)[D], std::int32_t device, Order order = kC)
      : order_{order} {
    auto &h_vec = data_.HostVector();
    h_vec.insert(h_vec.begin(), begin, end);
    // shape
    this->Initialize(shape, device);
  }

  template <typename I, int32_t D>
  explicit Tensor(std::initializer_list<T> data, I const (&shape)[D], std::int32_t device,
                  Order order = kC)
      : order_{order} {
    auto &h_vec = data_.HostVector();
    h_vec = data;
    // shape
    this->Initialize(shape, device);
  }
  /**
   * \brief Index operator. Not thread safe, should not be used in performance critical
   *        region. For more efficient indexing, consider getting a view first.
   */
  template <typename... Index>
  T &operator()(Index &&...idx) {
    return this->HostView()(std::forward<Index>(idx)...);
  }
  /**
   * \brief Index operator. Not thread safe, should not be used in performance critical
   *        region. For more efficient indexing, consider getting a view first.
   */
  template <typename... Index>
  T const &operator()(Index &&...idx) const {
    return this->HostView()(std::forward<Index>(idx)...);
  }

  /**
   * \brief Get a \ref TensorView for this tensor.
   */
  TensorView<T, kDim> View(int32_t device) {
    if (device >= 0) {
      data_.SetDevice(device);
      auto span = data_.DeviceSpan();
      return {span, shape_, device, order_};
    } else {
      auto span = data_.HostSpan();
      return {span, shape_, device, order_};
    }
  }
  TensorView<T const, kDim> View(int32_t device) const {
    if (device >= 0) {
      data_.SetDevice(device);
      auto span = data_.ConstDeviceSpan();
      return {span, shape_, device, order_};
    } else {
      auto span = data_.ConstHostSpan();
      return {span, shape_, device, order_};
    }
  }

  auto HostView() const { return this->View(-1); }
  auto HostView() { return this->View(-1); }

  [[nodiscard]] size_t Size() const { return data_.Size(); }
  auto Shape() const { return common::Span<size_t const, kDim>{shape_}; }
  auto Shape(size_t i) const { return shape_[i]; }

  HostDeviceVector<T> *Data() { return &data_; }
  HostDeviceVector<T> const *Data() const { return &data_; }

  /**
   * \brief Visitor function for modification that changes shape and data.
   *
   * \tparam Fn function that takes a pointer to `HostDeviceVector` and a static sized
   *         span as parameters.
   */
  template <typename Fn>
  void ModifyInplace(Fn &&fn) {
    fn(this->Data(), common::Span<size_t, kDim>{this->shape_});
    CHECK_EQ(this->Data()->Size(), detail::CalcSize(this->shape_))
        << "Inconsistent size after modification.";
  }

  /**
   * \brief Reshape the tensor.
   *
   *    If the total size is changed, then data in this tensor is no longer valid.
   */
  template <typename... S, detail::EnableIfIntegral<S...> * = nullptr>
  void Reshape(S &&...s) {
    static_assert(sizeof...(S) <= kDim, "Invalid shape.");
    detail::ReshapeImpl<0>(shape_, std::forward<S>(s)...);
    auto constexpr kEnd = sizeof...(S);
    static_assert(kEnd <= kDim, "Invalid shape.");
    std::fill(shape_ + kEnd, shape_ + kDim, 1);
    auto n = detail::CalcSize(shape_);
    data_.Resize(n);
  }

  /**
   * \brief Reshape the tensor.
   *
   *    If the total size is changed, then data in this tensor is no longer valid.
   */
  template <size_t D>
  void Reshape(common::Span<size_t const, D> shape) {
    static_assert(D <= kDim, "Invalid shape.");
    std::copy(shape.data(), shape.data() + D, this->shape_);
    std::fill(shape_ + D, shape_ + kDim, 1);
    auto n = detail::CalcSize(shape_);
    data_.Resize(n);
  }

  template <size_t D>
  void Reshape(size_t (&shape)[D]) {
    this->Reshape(common::Span<size_t const, D>{shape});
  }
  /**
   * \brief Get a host view on the slice.
   */
  template <typename... S>
  auto Slice(S &&...slices) const {
    return this->HostView().Slice(std::forward<S>(slices)...);
  }
  /**
   * \brief Get a host view on the slice.
   */
  template <typename... S>
  auto Slice(S &&...slices) {
    return this->HostView().Slice(std::forward<S>(slices)...);
  }

  /**
   * \brief Set device ordinal for this tensor.
   */
  void SetDevice(int32_t device) const { data_.SetDevice(device); }
  [[nodiscard]] int32_t DeviceIdx() const { return data_.DeviceIdx(); }
};

template <typename T>
using Matrix = Tensor<T, 2>;

template <typename T>
using Vector = Tensor<T, 1>;

/**
 * \brief Create an array without initialization.
 */
template <typename T, typename... Index>
auto Empty(Context const *ctx, Index &&...index) {
  Tensor<T, sizeof...(Index)> t;
  t.SetDevice(ctx->gpu_id);
  t.Reshape(index...);
  return t;
}

/**
 * \brief Create an array with value v.
 */
template <typename T, typename... Index>
auto Constant(Context const *ctx, T v, Index &&...index) {
  Tensor<T, sizeof...(Index)> t;
  t.SetDevice(ctx->gpu_id);
  t.Reshape(index...);
  t.Data()->Fill(std::move(v));
  return t;
}

/**
 * \brief Like `np.zeros`, return a new array of given shape and type, filled with zeros.
 */
template <typename T, typename... Index>
auto Zeros(Context const *ctx, Index &&...index) {
  return Constant(ctx, static_cast<T>(0), index...);
}

// Only first axis is supported for now.
template <typename T, int32_t D>
void Stack(Tensor<T, D> *l, Tensor<T, D> const &r) {
  if (r.DeviceIdx() >= 0) {
    l->SetDevice(r.DeviceIdx());
  }
  l->ModifyInplace([&](HostDeviceVector<T> *data, common::Span<size_t, D> shape) {
    for (size_t i = 1; i < D; ++i) {
      if (shape[i] == 0) {
        shape[i] = r.Shape(i);
      } else {
        CHECK_EQ(shape[i], r.Shape(i));
      }
    }
    data->Extend(*r.Data());
    shape[0] = l->Shape(0) + r.Shape(0);
  });
}
}  // namespace xgboost::linalg

#if defined(LINALG_HD)
#undef LINALG_HD
#endif  // defined(LINALG_HD)
#endif  // XGBOOST_LINALG_H_
