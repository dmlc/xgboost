/**
 * Copyright 2019-2023 by XGBoost Contributors
 * \file array_interface.h
 * \brief View of __array_interface__
 */
#ifndef XGBOOST_DATA_ARRAY_INTERFACE_H_
#define XGBOOST_DATA_ARRAY_INTERFACE_H_

#include <algorithm>
#include <cstddef>  // for size_t
#include <cstdint>
#include <limits>  // for numeric_limits
#include <map>
#include <string>
#include <type_traits>  // std::alignment_of,std::remove_pointer_t
#include <utility>
#include <vector>

#include "../common/bitfield.h"
#include "../common/common.h"
#include "../common/error_msg.h"  // for NoF128
#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/json.h"
#include "xgboost/linalg.h"
#include "xgboost/logging.h"
#include "xgboost/span.h"

namespace xgboost {
// Common errors in parsing columnar format.
struct ArrayInterfaceErrors {
  static char const *Contiguous() { return "Memory should be contiguous."; }
  static char const *TypestrFormat() {
    return "`typestr' should be of format <endian><type><size of type in bytes>.";
  }
  static char const *Dimension(int32_t d) {
    static std::string str;
    str.clear();
    str += "Only ";
    str += std::to_string(d);
    str += " dimensional array is valid.";
    return str.c_str();
  }
  static char const *Version() {
    return "Only version <= 3 of `__cuda_array_interface__' and `__array_interface__' are "
           "supported.";
  }
  static char const *OfType(std::string const &type) {
    static std::string str;
    str.clear();
    str += " should be of ";
    str += type;
    str += " type.";
    return str.c_str();
  }

  static std::string TypeStr(char c) {
    switch (c) {
      case 't':
        return "Bit field";
      case 'b':
        return "Boolean";
      case 'i':
        return "Integer";
      case 'u':
        return "Unsigned integer";
      case 'f':
        return "Floating point";
      case 'c':
        return "Complex floating point";
      case 'm':
        return "Timedelta";
      case 'M':
        return "Datetime";
      case 'O':
        return "Object";
      case 'S':
        return "String";
      case 'U':
        return "Unicode";
      case 'V':
        return "Other";
      default:
        LOG(FATAL) << "Invalid type code: " << c << " in `typestr' of input array."
                   << "\nPlease verify the `__cuda_array_interface__/__array_interface__' "
                   << "of your input data complies to: "
                   << "https://docs.scipy.org/doc/numpy/reference/arrays.interface.html"
                   << "\nOr open an issue.";
        return "";
    }
  }

  static std::string UnSupportedType(StringView typestr) {
    return TypeStr(typestr[1]) + "-" + typestr[2] + " is not supported.";
  }
};

/**
 * Utilities for consuming array interface.
 */
class ArrayInterfaceHandler {
 public:
  enum Type : std::int8_t { kF2, kF4, kF8, kF16, kI1, kI2, kI4, kI8, kU1, kU2, kU4, kU8 };

  template <typename PtrType>
  static PtrType GetPtrFromArrayData(Object::Map const &obj) {
    auto data_it = obj.find("data");
    if (data_it == obj.cend() || IsA<Null>(data_it->second)) {
      LOG(FATAL) << "Empty data passed in.";
    }
    auto p_data = reinterpret_cast<PtrType>(
        static_cast<size_t>(get<Integer const>(get<Array const>(data_it->second).at(0))));
    return p_data;
  }

  static void Validate(Object::Map const &array) {
    auto version_it = array.find("version");
    if (version_it == array.cend() || IsA<Null>(version_it->second)) {
      LOG(FATAL) << "Missing `version' field for array interface";
    }
    if (get<Integer const>(version_it->second) > 3) {
      LOG(FATAL) << ArrayInterfaceErrors::Version();
    }

    auto typestr_it = array.find("typestr");
    if (typestr_it == array.cend() || IsA<Null>(typestr_it->second)) {
      LOG(FATAL) << "Missing `typestr' field for array interface";
    }

    auto typestr = get<String const>(typestr_it->second);
    CHECK(typestr.size() == 3 || typestr.size() == 4) << ArrayInterfaceErrors::TypestrFormat();

    auto shape_it = array.find("shape");
    if (shape_it == array.cend() || IsA<Null>(shape_it->second)) {
      LOG(FATAL) << "Missing `shape' field for array interface";
    }
    auto data_it = array.find("data");
    if (data_it == array.cend() || IsA<Null>(data_it->second)) {
      LOG(FATAL) << "Missing `data' field for array interface";
    }
  }

  // Find null mask (validity mask) field
  // Mask object is also an array interface, but with different requirements.
  static size_t ExtractMask(Object::Map const &column,
                            common::Span<RBitField8::value_type> *p_out) {
    auto &s_mask = *p_out;
    auto const &mask_it = column.find("mask");
    if (mask_it != column.cend() && !IsA<Null>(mask_it->second)) {
      auto const &j_mask = get<Object const>(mask_it->second);
      Validate(j_mask);

      auto p_mask = GetPtrFromArrayData<RBitField8::value_type *>(j_mask);

      auto j_shape = get<Array const>(j_mask.at("shape"));
      CHECK_EQ(j_shape.size(), 1) << ArrayInterfaceErrors::Dimension(1);
      auto typestr = get<String const>(j_mask.at("typestr"));
      // For now this is just 1, we can support different size of interger in mask.
      int64_t const type_length = typestr.at(2) - 48;

      if (typestr.at(1) == 't') {
        CHECK_EQ(type_length, 1) << "mask with bitfield type should be of 1 byte per bitfield.";
      } else if (typestr.at(1) == 'i') {
        CHECK_EQ(type_length, 1) << "mask with integer type should be of 1 byte per integer.";
      } else {
        LOG(FATAL) << "mask must be of integer type or bit field type.";
      }
      /*
       * shape represents how many bits is in the mask. (This is a grey area, don't be
       * suprised if it suddently represents something else when supporting a new
       * implementation).  Quoting from numpy array interface:
       *
       *   The shape of this object should be "broadcastable" to the shape of the original
       *   array.
       *
       * And that's the only requirement.
       */
      size_t const n_bits = static_cast<size_t>(get<Integer>(j_shape.at(0)));
      // The size of span required to cover all bits.  Here with 8 bits bitfield, we
      // assume 1 byte alignment.
      size_t const span_size = RBitField8::ComputeStorageSize(n_bits);

      auto strides_it = j_mask.find("strides");
      if (strides_it != j_mask.cend() && !IsA<Null>(strides_it->second)) {
        auto strides = get<Array const>(strides_it->second);
        CHECK_EQ(strides.size(), 1) << ArrayInterfaceErrors::Dimension(1);
        CHECK_EQ(get<Integer>(strides.at(0)), type_length) << ArrayInterfaceErrors::Contiguous();
      }

      s_mask = {p_mask, span_size};
      return n_bits;
    }
    return 0;
  }
  /**
   * \brief Handle vector inputs.  For higher dimension, we require strictly correct shape.
   */
  template <int32_t D>
  static void HandleRowVector(std::vector<size_t> const &shape, std::vector<size_t> *p_out) {
    auto &out = *p_out;
    if (shape.size() == 2 && D == 1) {
      auto m = shape[0];
      auto n = shape[1];
      CHECK(m == 1 || n == 1);
      if (m == 1) {
        // keep the number of columns
        out[0] = out[1];
        out.resize(1);
      } else if (n == 1) {
        // keep the number of rows.
        out.resize(1);
      }
      // when both m and n are 1, above logic keeps the column.
      // when neither m nor n is 1, caller should throw an error about Dimension.
    }
  }

  template <int32_t D>
  static void ExtractShape(Object::Map const &array, size_t (&out_shape)[D]) {
    auto const &j_shape = get<Array const>(array.at("shape"));
    std::vector<size_t> shape_arr(j_shape.size(), 0);
    std::transform(j_shape.cbegin(), j_shape.cend(), shape_arr.begin(),
                   [](Json in) { return get<Integer const>(in); });
    // handle column vector vs. row vector
    HandleRowVector<D>(shape_arr, &shape_arr);
    // Copy shape.
    size_t i;
    for (i = 0; i < shape_arr.size(); ++i) {
      CHECK_LT(i, D) << ArrayInterfaceErrors::Dimension(D);
      out_shape[i] = shape_arr[i];
    }
    // Fill the remaining dimensions
    std::fill(out_shape + i, out_shape + D, 1);
  }

  /**
   * \brief Extracts the optiona `strides' field and returns whether the array is c-contiguous.
   */
  template <int32_t D>
  static bool ExtractStride(Object::Map const &array, size_t itemsize,
                            size_t (&shape)[D], size_t (&stride)[D]) {
    auto strides_it = array.find("strides");
    // No stride is provided
    if (strides_it == array.cend() || IsA<Null>(strides_it->second)) {
      // No stride is provided, we can calculate it from shape.
      linalg::detail::CalcStride(shape, stride);
      // Quote:
      //
      //   strides: Either None to indicate a C-style contiguous array or a Tuple of
      //            strides which provides the number of bytes
      return true;
    }
    // Get shape, we need to make changes to handle row vector, so some duplicated code
    // from `ExtractShape` for copying out the shape.
    auto const &j_shape = get<Array const>(array.at("shape"));
    std::vector<size_t> shape_arr(j_shape.size(), 0);
    std::transform(j_shape.cbegin(), j_shape.cend(), shape_arr.begin(),
                   [](Json in) { return get<Integer const>(in); });
    // Get stride
    auto const &j_strides = get<Array const>(strides_it->second);
    CHECK_EQ(j_strides.size(), j_shape.size()) << "stride and shape don't match.";
    std::vector<size_t> stride_arr(j_strides.size(), 0);
    std::transform(j_strides.cbegin(), j_strides.cend(), stride_arr.begin(),
                   [](Json in) { return get<Integer const>(in); });

    // Handle column vector vs. row vector
    HandleRowVector<D>(shape_arr, &stride_arr);
    size_t i;
    for (i = 0; i < stride_arr.size(); ++i) {
      // If one of the dim has shape 0 then total size is 0, stride is meaningless, but we
      // set it to 0 here just to be consistent
      CHECK_LT(i, D) << ArrayInterfaceErrors::Dimension(D);
      // We use number of items instead of number of bytes
      stride[i] = stride_arr[i] / itemsize;
    }
    std::fill(stride + i, stride + D, 1);
    // If the stride can be calculated from shape then it's contiguous.
    size_t stride_tmp[D];
    linalg::detail::CalcStride(shape, stride_tmp);
    return std::equal(stride_tmp, stride_tmp + D, stride);
  }

  static void *ExtractData(Object::Map const &array, size_t size) {
    Validate(array);
    void *p_data = ArrayInterfaceHandler::GetPtrFromArrayData<void *>(array);
    if (!p_data) {
      CHECK_EQ(size, 0) << "Empty data with non-zero shape.";
    }
    return p_data;
  }
  /**
   * \brief Whether the ptr is allocated by CUDA.
   */
  static bool IsCudaPtr(void const *ptr);
  /**
   * \brief Sync the CUDA stream.
   */
  static void SyncCudaStream(int64_t stream);
};

/**
 * Dispatch compile time type to runtime type.
 */
template <typename T, typename E = void>
struct ToDType;
// float
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
template <>
struct ToDType<__half> {
  static constexpr ArrayInterfaceHandler::Type kType = ArrayInterfaceHandler::kF2;
};
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
template <>
struct ToDType<float> {
  static constexpr ArrayInterfaceHandler::Type kType = ArrayInterfaceHandler::kF4;
};
template <>
struct ToDType<double> {
  static constexpr ArrayInterfaceHandler::Type kType = ArrayInterfaceHandler::kF8;
};
template <typename T>
struct ToDType<T,
               std::enable_if_t<std::is_same<T, long double>::value && sizeof(long double) == 16>> {
  static constexpr ArrayInterfaceHandler::Type kType = ArrayInterfaceHandler::kF16;
};
// uint
template <>
struct ToDType<uint8_t> {
  static constexpr ArrayInterfaceHandler::Type kType = ArrayInterfaceHandler::kU1;
};
template <>
struct ToDType<uint16_t> {
  static constexpr ArrayInterfaceHandler::Type kType = ArrayInterfaceHandler::kU2;
};
template <>
struct ToDType<uint32_t> {
  static constexpr ArrayInterfaceHandler::Type kType = ArrayInterfaceHandler::kU4;
};
template <>
struct ToDType<uint64_t> {
  static constexpr ArrayInterfaceHandler::Type kType = ArrayInterfaceHandler::kU8;
};
// int
template <>
struct ToDType<int8_t> {
  static constexpr ArrayInterfaceHandler::Type kType = ArrayInterfaceHandler::kI1;
};
template <>
struct ToDType<int16_t> {
  static constexpr ArrayInterfaceHandler::Type kType = ArrayInterfaceHandler::kI2;
};
template <>
struct ToDType<int32_t> {
  static constexpr ArrayInterfaceHandler::Type kType = ArrayInterfaceHandler::kI4;
};
template <>
struct ToDType<int64_t> {
  static constexpr ArrayInterfaceHandler::Type kType = ArrayInterfaceHandler::kI8;
};

#if !defined(XGBOOST_USE_CUDA)
inline void ArrayInterfaceHandler::SyncCudaStream(int64_t) { common::AssertGPUSupport(); }
inline bool ArrayInterfaceHandler::IsCudaPtr(void const *) { return false; }
#endif  // !defined(XGBOOST_USE_CUDA)

/**
 * \brief A type erased view over __array_interface__ protocol defined by numpy
 *
 *   <a href="https://numpy.org/doc/stable/reference/arrays.interface.html">numpy</a>.
 *
 * \tparam D The number of maximum dimension.

 *   User input array must have dim <= D for all non-trivial dimensions.  During
 *   construction, the ctor can automatically remove those trivial dimensions.
 *
 * \tparam allow_mask Whether masked array is accepted.
 *
 *   Currently this only supported for 1-dim vector, which is used by cuDF column
 *   (apache arrow format).  For general masked array, as the time of writting, only
 *   numpy has the proper support even though it's in the __cuda_array_interface__
 *   protocol defined by numba.
 */
template <int32_t D, bool allow_mask = (D == 1)>
class ArrayInterface {
  static_assert(D > 0, "Invalid dimension for array interface.");

  /**
   * \brief Initialize the object, by extracting shape, stride and type.
   *
   *   The function also perform some basic validation for input array.  Lastly it will
   *   also remove trivial dimensions like converting a matrix with shape (n_samples, 1)
   *   to a vector of size n_samples.  For for inputs like weights, this should be a 1
   *   dimension column vector even though user might provide a matrix.
   */
  void Initialize(Object::Map const &array) {
    ArrayInterfaceHandler::Validate(array);

    auto typestr = get<String const>(array.at("typestr"));
    this->AssignType(StringView{typestr});
    ArrayInterfaceHandler::ExtractShape(array, shape);
    size_t itemsize = typestr[2] - '0';
    is_contiguous = ArrayInterfaceHandler::ExtractStride(array, itemsize, shape, strides);
    n = linalg::detail::CalcSize(shape);

    data = ArrayInterfaceHandler::ExtractData(array, n);
    static_assert(allow_mask ? D == 1 : D >= 1, "Masked ndarray is not supported.");

    auto alignment = this->ElementAlignment();
    auto ptr = reinterpret_cast<uintptr_t>(this->data);
    CHECK_EQ(ptr % alignment, 0) << "Input pointer misalignment.";

    if (allow_mask) {
      common::Span<RBitField8::value_type> s_mask;
      size_t n_bits = ArrayInterfaceHandler::ExtractMask(array, &s_mask);

      valid = RBitField8(s_mask);

      if (s_mask.data()) {
        CHECK_EQ(n_bits, n) << "Shape of bit mask doesn't match data shape. "
                            << "XGBoost doesn't support internal broadcasting.";
      }
    } else {
      auto mask_it = array.find("mask");
      CHECK(mask_it == array.cend() || IsA<Null>(mask_it->second))
          << "Masked array is not yet supported.";
    }

    auto stream_it = array.find("stream");
    if (stream_it != array.cend() && !IsA<Null>(stream_it->second)) {
      int64_t stream = get<Integer const>(stream_it->second);
      ArrayInterfaceHandler::SyncCudaStream(stream);
    }
  }

 public:
  ArrayInterface() = default;
  explicit ArrayInterface(Object::Map const &array) { this->Initialize(array); }

  explicit ArrayInterface(Json const &array) {
    if (IsA<Object>(array)) {
      this->Initialize(get<Object const>(array));
      return;
    }
    if (IsA<Array>(array)) {
      CHECK_EQ(get<Array const>(array).size(), 1)
          << "Column: " << ArrayInterfaceErrors::Dimension(1);
      this->Initialize(get<Object const>(get<Array const>(array)[0]));
      return;
    }
  }

  explicit ArrayInterface(std::string const &str) : ArrayInterface{StringView{str}} {}

  explicit ArrayInterface(StringView str) : ArrayInterface<D>{Json::Load(str)} {}

  void AssignType(StringView typestr) {
    using T = ArrayInterfaceHandler::Type;
    if (typestr.size() == 4 && typestr[1] == 'f' && typestr[2] == '1' && typestr[3] == '6') {
      CHECK(sizeof(long double) == 16) << error::NoF128();
      type = T::kF16;
    } else if (typestr[1] == 'f' && typestr[2] == '2') {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
      type = T::kF2;
#else
      LOG(FATAL) << "Half type is not supported.";
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
    } else if (typestr[1] == 'f' && typestr[2] == '4') {
      type = T::kF4;
    } else if (typestr[1] == 'f' && typestr[2] == '8') {
      type = T::kF8;
    } else if (typestr[1] == 'i' && typestr[2] == '1') {
      type = T::kI1;
    } else if (typestr[1] == 'i' && typestr[2] == '2') {
      type = T::kI2;
    } else if (typestr[1] == 'i' && typestr[2] == '4') {
      type = T::kI4;
    } else if (typestr[1] == 'i' && typestr[2] == '8') {
      type = T::kI8;
    } else if (typestr[1] == 'u' && typestr[2] == '1') {
      type = T::kU1;
    } else if (typestr[1] == 'u' && typestr[2] == '2') {
      type = T::kU2;
    } else if (typestr[1] == 'u' && typestr[2] == '4') {
      type = T::kU4;
    } else if (typestr[1] == 'u' && typestr[2] == '8') {
      type = T::kU8;
    } else {
      LOG(FATAL) << ArrayInterfaceErrors::UnSupportedType(typestr);
      return;
    }
  }

  XGBOOST_DEVICE size_t Shape(size_t i) const { return shape[i]; }
  XGBOOST_DEVICE size_t Stride(size_t i) const { return strides[i]; }

  template <typename Fn>
  XGBOOST_HOST_DEV_INLINE decltype(auto) DispatchCall(Fn func) const {
    using T = ArrayInterfaceHandler::Type;
    switch (type) {
      case T::kF2: {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
        return func(reinterpret_cast<__half const *>(data));
#else
        SPAN_CHECK(false);
        return func(reinterpret_cast<float const *>(data));
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
      }
      case T::kF4:
        return func(reinterpret_cast<float const *>(data));
      case T::kF8:
        return func(reinterpret_cast<double const *>(data));
#ifdef __CUDA_ARCH__
      case T::kF16: {
        // CUDA device code doesn't support long double.
        SPAN_CHECK(false);
        return func(reinterpret_cast<double const *>(data));
      }
#else
      case T::kF16:
        return func(reinterpret_cast<long double const *>(data));
#endif
      case T::kI1:
        return func(reinterpret_cast<int8_t const *>(data));
      case T::kI2:
        return func(reinterpret_cast<int16_t const *>(data));
      case T::kI4:
        return func(reinterpret_cast<int32_t const *>(data));
      case T::kI8:
        return func(reinterpret_cast<int64_t const *>(data));
      case T::kU1:
        return func(reinterpret_cast<uint8_t const *>(data));
      case T::kU2:
        return func(reinterpret_cast<uint16_t const *>(data));
      case T::kU4:
        return func(reinterpret_cast<uint32_t const *>(data));
      case T::kU8:
        return func(reinterpret_cast<uint64_t const *>(data));
    }
    SPAN_CHECK(false);
    return func(reinterpret_cast<uint64_t const *>(data));
  }

  XGBOOST_DEVICE std::size_t ElementSize() const {
    return this->DispatchCall([](auto *typed_data_ptr) {
      return sizeof(std::remove_pointer_t<decltype(typed_data_ptr)>);
    });
  }
  XGBOOST_DEVICE std::size_t ElementAlignment() const {
    return this->DispatchCall([](auto *typed_data_ptr) {
      return std::alignment_of<std::remove_pointer_t<decltype(typed_data_ptr)>>::value;
    });
  }

  template <typename T = float, typename... Index>
  XGBOOST_DEVICE T operator()(Index &&...index) const {
    static_assert(sizeof...(index) <= D, "Invalid index.");
    return this->DispatchCall([=](auto const *p_values) -> T {
      std::size_t offset = linalg::detail::Offset<0ul>(strides, 0ul, index...);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
      // No operator defined for half -> size_t
      using Type = std::conditional_t<
          std::is_same<__half,
                       std::remove_cv_t<std::remove_pointer_t<decltype(p_values)>>>::value &&
              std::is_same<std::size_t, std::remove_cv_t<T>>::value,
          unsigned long long, T>;  // NOLINT
      return static_cast<T>(static_cast<Type>(p_values[offset]));
#else
      return static_cast<T>(p_values[offset]);
#endif
    });
  }

  // Used only by columnar format.
  RBitField8 valid;
  // Array stride
  std::size_t strides[D]{0};
  // Array shape
  std::size_t shape[D]{0};
  // Type earsed pointer referencing the data.
  void const *data{nullptr};
  // Total number of items
  std::size_t n{0};
  // Whether the memory is c-contiguous
  bool is_contiguous{false};
  // RTTI, initialized to the f16 to avoid masking potential bugs in initialization.
  ArrayInterfaceHandler::Type type{ArrayInterfaceHandler::kF16};
};

template <std::int32_t D, typename Fn>
void DispatchDType(ArrayInterface<D> const array, std::int32_t device, Fn fn) {
  // Only used for cuDF at the moment.
  CHECK_EQ(array.valid.Size(), 0);
  auto dispatch = [&](auto t) {
    using T = std::remove_const_t<decltype(t)> const;
    // Set the data size to max as we don't know the original size of a sliced array:
    //
    // Slicing an array A with shape (4, 2, 3) and stride (6, 3, 1) by [:, 1, :] results
    // in an array B with shape (4, 3) and strides (6, 1). We can't calculate the original
    // size 24 based on the slice.
    fn(linalg::TensorView<T, D>{common::Span<T const>{static_cast<T *>(array.data),
                                                      std::numeric_limits<std::size_t>::max()},
                                array.shape, array.strides, device});
  };
  switch (array.type) {
    case ArrayInterfaceHandler::kF2: {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
      dispatch(__half{});
#endif
      break;
    }
    case ArrayInterfaceHandler::kF4: {
      dispatch(float{});
      break;
    }
    case ArrayInterfaceHandler::kF8: {
      dispatch(double{});
      break;
    }
    case ArrayInterfaceHandler::kF16: {
      using T = long double;
      CHECK(sizeof(long double) == 16) << error::NoF128();
      dispatch(T{});
      break;
    }
    case ArrayInterfaceHandler::kI1: {
      dispatch(std::int8_t{});
      break;
    }
    case ArrayInterfaceHandler::kI2: {
      dispatch(std::int16_t{});
      break;
    }
    case ArrayInterfaceHandler::kI4: {
      dispatch(std::int32_t{});
      break;
    }
    case ArrayInterfaceHandler::kI8: {
      dispatch(std::int64_t{});
      break;
    }
    case ArrayInterfaceHandler::kU1: {
      dispatch(std::uint8_t{});
      break;
    }
    case ArrayInterfaceHandler::kU2: {
      dispatch(std::uint16_t{});
      break;
    }
    case ArrayInterfaceHandler::kU4: {
      dispatch(std::uint32_t{});
      break;
    }
    case ArrayInterfaceHandler::kU8: {
      dispatch(std::uint64_t{});
      break;
    }
  }
}

/**
 * \brief Helper for type casting.
 */
template <typename T, int32_t D>
struct TypedIndex {
  ArrayInterface<D> const &array;
  template <typename... I>
  XGBOOST_DEVICE T operator()(I &&...ind) const {
    static_assert(sizeof...(ind) <= D, "Invalid index.");
    return array.template operator()<T>(ind...);
  }
};

template <int32_t D>
inline void CheckArrayInterface(StringView key, ArrayInterface<D> const &array) {
  CHECK(!array.valid.Data()) << "Meta info " << key << " should be dense, found validity mask";
}
}  // namespace xgboost
#endif  // XGBOOST_DATA_ARRAY_INTERFACE_H_
