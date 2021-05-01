/*!
 * Copyright 2019-2021 by Contributors
 * \file array_interface.h
 * \brief View of __array_interface__
 */
#ifndef XGBOOST_DATA_ARRAY_INTERFACE_H_
#define XGBOOST_DATA_ARRAY_INTERFACE_H_

#include <algorithm>
#include <cinttypes>
#include <map>
#include <string>
#include <utility>

#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/json.h"
#include "xgboost/logging.h"
#include "xgboost/span.h"
#include "../common/bitfield.h"
#include "../common/common.h"

namespace xgboost {
// Common errors in parsing columnar format.
struct ArrayInterfaceErrors {
  static char const* Contigious() {
    return "Memory should be contigious.";
  }
  static char const* TypestrFormat() {
    return "`typestr' should be of format <endian><type><size of type in bytes>.";
  }
  // Not supported in Apache Arrow.
  static char const* BigEndian() {
    return "Big endian is not supported.";
  }
  static char const* Dimension(int32_t d) {
    static std::string str;
    str.clear();
    str += "Only ";
    str += std::to_string(d);
    str += " dimensional array is valid.";
    return str.c_str();
  }
  static char const* Version() {
    return "Only version <= 3 of `__cuda_array_interface__' are supported.";
  }
  static char const* OfType(std::string const& type) {
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
                   << "\nPlease verify the `__cuda_array_interface__' "
                   << "of your input data complies to: "
                   << "https://docs.scipy.org/doc/numpy/reference/arrays.interface.html"
                   << "\nOr open an issue.";
        return "";
    }
  }

  static std::string UnSupportedType(StringView typestr) {
    return TypeStr(typestr[1]) + " is not supported.";
  }
};

// TODO(trivialfis): Abstract this into a class that accept a json
// object and turn it into an array (for cupy and numba).
class ArrayInterfaceHandler {
 public:
  template <typename T>
  static constexpr char TypeChar() {
    return
        (std::is_floating_point<T>::value ? 'f' :
         (std::is_integral<T>::value ?
          (std::is_signed<T>::value ? 'i' : 'u') : '\0'));
  }

  template <typename PtrType>
  static PtrType GetPtrFromArrayData(std::map<std::string, Json> const& obj) {
    if (obj.find("data") == obj.cend()) {
      LOG(FATAL) << "Empty data passed in.";
    }
    auto p_data = reinterpret_cast<PtrType>(static_cast<size_t>(
        get<Integer const>(
            get<Array const>(
                obj.at("data"))
            .at(0))));
    return p_data;
  }

  static void Validate(std::map<std::string, Json> const& array) {
    auto version_it = array.find("version");
    if (version_it == array.cend()) {
      LOG(FATAL) << "Missing `version' field for array interface";
    }
    auto stream_it = array.find("stream");
    if (stream_it != array.cend() && !IsA<Null>(stream_it->second)) {
      // is cuda, check the version.
      if (get<Integer const>(version_it->second) > 3) {
        LOG(FATAL) << ArrayInterfaceErrors::Version();
      }
    }

    if (array.find("typestr") == array.cend()) {
      LOG(FATAL) << "Missing `typestr' field for array interface";
    }
    auto typestr = get<String const>(array.at("typestr"));
    CHECK_EQ(typestr.size(),    3) << ArrayInterfaceErrors::TypestrFormat();
    CHECK_NE(typestr.front(), '>') << ArrayInterfaceErrors::BigEndian();

    if (array.find("shape") == array.cend()) {
      LOG(FATAL) << "Missing `shape' field for array interface";
    }
    if (array.find("data") == array.cend()) {
      LOG(FATAL) << "Missing `data' field for array interface";
    }
  }

  // Find null mask (validity mask) field
  // Mask object is also an array interface, but with different requirements.
  static size_t ExtractMask(std::map<std::string, Json> const &column,
                            common::Span<RBitField8::value_type> *p_out) {
    auto& s_mask = *p_out;
    if (column.find("mask") != column.cend()) {
      auto const& j_mask = get<Object const>(column.at("mask"));
      Validate(j_mask);

      auto p_mask = GetPtrFromArrayData<RBitField8::value_type*>(j_mask);

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

      if (j_mask.find("strides") != j_mask.cend()) {
        auto strides = get<Array const>(column.at("strides"));
        CHECK_EQ(strides.size(),                        1) << ArrayInterfaceErrors::Dimension(1);
        CHECK_EQ(get<Integer>(strides.at(0)), type_length) << ArrayInterfaceErrors::Contigious();
      }

      s_mask = {p_mask, span_size};
      return n_bits;
    }
    return 0;
  }

  static std::pair<bst_row_t, bst_feature_t> ExtractShape(
      std::map<std::string, Json> const& column) {
    auto j_shape = get<Array const>(column.at("shape"));
    auto typestr = get<String const>(column.at("typestr"));
    if (j_shape.size() == 1) {
      return {static_cast<bst_row_t>(get<Integer const>(j_shape.at(0))), 1};
    } else {
      CHECK_EQ(j_shape.size(), 2) << "Only 1-D and 2-D arrays are supported.";
      return {static_cast<bst_row_t>(get<Integer const>(j_shape.at(0))),
              static_cast<bst_feature_t>(get<Integer const>(j_shape.at(1)))};
    }
  }

  static void ExtractStride(std::map<std::string, Json> const &column,
                            size_t strides[2], size_t rows, size_t cols, size_t itemsize) {
    auto strides_it = column.find("strides");
    if (strides_it == column.cend() || IsA<Null>(strides_it->second)) {
      // default strides
      strides[0] = cols;
      strides[1] = 1;
    } else {
      // strides specified by the array interface
      auto const &j_strides = get<Array const>(strides_it->second);
      CHECK_LE(j_strides.size(), 2) << ArrayInterfaceErrors::Dimension(2);
      strides[0] = get<Integer const>(j_strides[0]) / itemsize;
      size_t n = 1;
      if (j_strides.size() == 2) {
        n = get<Integer const>(j_strides[1]) / itemsize;
      }
      strides[1] = n;
    }

    auto valid = rows * strides[0] + cols * strides[1] >= (rows * cols);
    CHECK(valid) << "Invalid strides in array."
                 << "  strides: (" << strides[0] << "," << strides[1]
                 << "), shape: (" << rows << ", " << cols << ")";
  }

  static void* ExtractData(std::map<std::string, Json> const &column,
                                     StringView typestr,
                                     std::pair<size_t, size_t> shape) {
    Validate(column);
    void* p_data = ArrayInterfaceHandler::GetPtrFromArrayData<void*>(column);
    if (!p_data) {
      CHECK_EQ(shape.first * shape.second, 0) << "Empty data with non-zero shape.";
    }
    return p_data;
  }

  static void SyncCudaStream(int64_t stream);
};

#if !defined(XGBOOST_USE_CUDA)
inline void ArrayInterfaceHandler::SyncCudaStream(int64_t stream) {
  common::AssertGPUSupport();
}
#endif  // !defined(XGBOOST_USE_CUDA)

// A view over __array_interface__
class ArrayInterface {
  void Initialize(std::map<std::string, Json> const &array,
                  bool allow_mask = true) {
    ArrayInterfaceHandler::Validate(array);
    auto typestr = get<String const>(array.at("typestr"));
    this->AssignType(StringView{typestr});

    std::tie(num_rows, num_cols) = ArrayInterfaceHandler::ExtractShape(array);
    data = ArrayInterfaceHandler::ExtractData(
        array, StringView{typestr}, std::make_pair(num_rows, num_cols));

    if (allow_mask) {
      common::Span<RBitField8::value_type> s_mask;
      size_t n_bits = ArrayInterfaceHandler::ExtractMask(array, &s_mask);

      valid = RBitField8(s_mask);

      if (s_mask.data()) {
        CHECK_EQ(n_bits, num_rows)
            << "Shape of bit mask doesn't match data shape. "
            << "XGBoost doesn't support internal broadcasting.";
      }
    } else {
      CHECK(array.find("mask") == array.cend())
          << "Masked array is not yet supported.";
    }

    ArrayInterfaceHandler::ExtractStride(array, strides, num_rows, num_cols,
                                         typestr[2] - '0');

    auto stream_it = array.find("stream");
    if (stream_it != array.cend() && !IsA<Null>(stream_it->second)) {
      int64_t stream = get<Integer const>(stream_it->second);
      ArrayInterfaceHandler::SyncCudaStream(stream);
    }
  }

 public:
  enum Type : std::int8_t { kF4, kF8, kI1, kI2, kI4, kI8, kU1, kU2, kU4, kU8 };

 public:
  ArrayInterface() = default;
  explicit ArrayInterface(std::string const &str, bool allow_mask = true)
      : ArrayInterface{StringView{str.c_str(), str.size()}, allow_mask} {}

  explicit ArrayInterface(std::map<std::string, Json> const &column,
                          bool allow_mask = true) {
    this->Initialize(column, allow_mask);
  }

  explicit ArrayInterface(StringView str, bool allow_mask = true) {
    auto jinterface = Json::Load(str);
    if (IsA<Object>(jinterface)) {
      this->Initialize(get<Object const>(jinterface), allow_mask);
      return;
    }
    if (IsA<Array>(jinterface)) {
      CHECK_EQ(get<Array const>(jinterface).size(), 1)
          << "Column: " << ArrayInterfaceErrors::Dimension(1);
      this->Initialize(get<Object const>(get<Array const>(jinterface)[0]), allow_mask);
      return;
    }
  }

  void AsColumnVector() {
    CHECK(num_rows == 1 || num_cols == 1) << "Array should be a vector instead of matrix.";
    num_rows = std::max(num_rows, static_cast<size_t>(num_cols));
    num_cols = 1;

    strides[0] = std::max(strides[0], strides[1]);
    strides[1] = 1;
  }

  void AssignType(StringView typestr) {
    if (typestr[1] == 'f' && typestr[2] == '4') {
      type = kF4;
    } else if (typestr[1] == 'f' && typestr[2] == '8') {
      type = kF8;
    } else if (typestr[1] == 'i' && typestr[2] == '1') {
      type = kI1;
    } else if (typestr[1] == 'i' && typestr[2] == '2') {
      type = kI2;
    } else if (typestr[1] == 'i' && typestr[2] == '4') {
      type = kI4;
    } else if (typestr[1] == 'i' && typestr[2] == '8') {
      type = kI8;
    } else if (typestr[1] == 'u' && typestr[2] == '1') {
      type = kU1;
    } else if (typestr[1] == 'u' && typestr[2] == '2') {
      type = kU2;
    } else if (typestr[1] == 'u' && typestr[2] == '4') {
      type = kU4;
    } else if (typestr[1] == 'u' && typestr[2] == '8') {
      type = kU8;
    } else {
      LOG(FATAL) << ArrayInterfaceErrors::UnSupportedType(typestr);
      return;
    }
  }

  template <typename Fn>
  XGBOOST_HOST_DEV_INLINE decltype(auto) DispatchCall(Fn func) const {
    switch (type) {
    case kF4:
      return func(reinterpret_cast<float *>(data));
    case kF8:
      return func(reinterpret_cast<double *>(data));
    case kI1:
      return func(reinterpret_cast<int8_t *>(data));
    case kI2:
      return func(reinterpret_cast<int16_t *>(data));
    case kI4:
      return func(reinterpret_cast<int32_t *>(data));
    case kI8:
      return func(reinterpret_cast<int64_t *>(data));
    case kU1:
      return func(reinterpret_cast<uint8_t *>(data));
    case kU2:
      return func(reinterpret_cast<uint16_t *>(data));
    case kU4:
      return func(reinterpret_cast<uint32_t *>(data));
    case kU8:
      return func(reinterpret_cast<uint64_t *>(data));
    }
    SPAN_CHECK(false);
    return func(reinterpret_cast<uint64_t *>(data));
  }

  XGBOOST_DEVICE size_t ElementSize() {
    return this->DispatchCall([](auto* p_values) {
      return sizeof(std::remove_pointer_t<decltype(p_values)>);
    });
  }

  template <typename T = float>
  XGBOOST_DEVICE T GetElement(size_t r, size_t c) const {
    return this->DispatchCall(
        [=](auto *p_values) -> T { return p_values[strides[0] * r + strides[1] * c]; });
  }

  RBitField8 valid;
  bst_row_t num_rows;
  bst_feature_t num_cols;
  size_t strides[2]{0, 0};
  void* data;
  Type type;
};

}  // namespace xgboost
#endif  // XGBOOST_DATA_ARRAY_INTERFACE_H_
