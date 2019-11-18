/*!
 * Copyright 2019 by Contributors
 * \file columnar.h
 * \brief Basic structure holding a reference to arrow columnar data format.
 */
#ifndef XGBOOST_DATA_COLUMNAR_H_
#define XGBOOST_DATA_COLUMNAR_H_

#include <cinttypes>
#include <map>
#include <string>

#include "xgboost/data.h"
#include "xgboost/logging.h"
#include "xgboost/span.h"

#include "../common/bitfield.h"
#include "../common/json_experimental.h"

namespace xgboost {
// A view over __array_interface__
template <typename T>
struct Columnar {
  using mask_type = unsigned char;
  using index_type = int32_t;

  common::Span<T>  data;
  RBitField8 valid;
  int32_t size;
};

// Common errors in parsing columnar format.
struct ColumnarErrors {
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
    return "Only version 1 of `__cuda_array_interface__' is supported.";
  }
  static char const* ofType(std::string const& type) {
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

  static std::string UnSupportedType(experimental::ConstStringRef typestr) {
    return TypeStr(typestr.at(1)) + " is not supported.";
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
  static PtrType GetPtrFromArrayData(experimental::Json const& obj) {
    if (obj.FindMemberByKey("data") == obj.cend()) {
      LOG(FATAL) << "Empty data passed in.";
    }
    auto p_data = reinterpret_cast<PtrType>(
        static_cast<size_t>(
            (*obj.FindMemberByKey("data"))
              .GetArrayElem(0)  // first element is pointer, second is flag for read-only
                .GetInt()));    // pointer as signed integer
    return p_data;
  }

  static void Validate(experimental::Json const& array) {
    if (array.FindMemberByKey("version") == array.cend()) {
      LOG(FATAL) << "Missing `version' field for array interface";
    }
    auto version = (*array.FindMemberByKey("version")).GetInt();
    CHECK_EQ(version, 1) << ColumnarErrors::Version();

    if (array.FindMemberByKey("typestr") == array.cend()) {
      LOG(FATAL) << "Missing `typestr' field for array interface";
    }
    auto typestr = (*array.FindMemberByKey("typestr")).GetString();
    CHECK_EQ(typestr.size(),    3) << ColumnarErrors::TypestrFormat();
    CHECK_NE(typestr.front(), '>') << ColumnarErrors::BigEndian();

    if (array.FindMemberByKey("shape") == array.cend()) {
      LOG(FATAL) << "Missing `shape' field for array interface";
    }
    if (array.FindMemberByKey("data") == array.cend()) {
      LOG(FATAL) << "Missing `data' field for array interface";
    }
  }

  // Find null mask (validity mask) field
  // Mask object is also an array interface, but with different requirements.
  static size_t ExtractMask(experimental::Json const &column,
                            common::Span<RBitField8::value_type> *p_out) {
    auto& s_mask = *p_out;
    if (column.FindMemberByKey("mask") != column.cend()) {
      auto j_mask = *column.FindMemberByKey("mask");
      Validate(j_mask);

      auto p_mask = GetPtrFromArrayData<RBitField8::value_type*>(j_mask);

      auto j_shape = *j_mask.FindMemberByKey("shape");
      CHECK_EQ(j_shape.Length(), 1) << ColumnarErrors::Dimension(1);
      auto typestr = (*j_mask.FindMemberByKey("typestr")).GetString();
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
      size_t const n_bits = static_cast<size_t>(j_shape.GetArrayElem(0).GetInt());
      // The size of span required to cover all bits.  Here with 8 bits bitfield, we
      // assume 1 byte alignment.
      size_t const span_size = RBitField8::ComputeStorageSize(n_bits);

      if (j_mask.FindMemberByKey("strides") != j_mask.cend()) {
        auto strides = *column.FindMemberByKey("strides");
        CHECK_EQ(strides.Length(),                           1) << ColumnarErrors::Dimension(1);
        CHECK_EQ(strides.GetArrayElem(0).GetInt(), type_length) << ColumnarErrors::Contigious();
      }

      s_mask = {p_mask, span_size};
      return n_bits;
    }
    return 0;
  }

  /*
   * \param column A JSON object representing a column.
   */
  template <typename T>
  static common::Span<T> ExtractData(experimental::Json const& column) {
    Validate(column);
    auto typestr = (*column.FindMemberByKey("typestr")).GetString();
    CHECK_EQ(typestr.at(1),   TypeChar<T>())
        << "Input data type and typestr mismatch. typestr: " << typestr;
    CHECK_EQ(typestr.at(2),   static_cast<char>(sizeof(T) + 48))
        << "Input data type and typestr mismatch. typestr: " << typestr;

    auto j_shape = *column.FindMemberByKey("shape");
    CHECK_EQ(j_shape.Length(), 1) << ColumnarErrors::Dimension(1);

    if (column.FindMemberByKey("strides") != column.cend()) {
      auto strides = *column.FindMemberByKey("strides");
      CHECK_EQ(strides.Length(),                         1) << ColumnarErrors::Dimension(1);
      CHECK_EQ(strides.GetArrayElem(0).GetInt(), sizeof(T)) << ColumnarErrors::Contigious();
    }

    auto length = static_cast<size_t>(j_shape.GetArrayElem(0).GetInt());

    T* p_data = ArrayInterfaceHandler::GetPtrFromArrayData<T*>(column);
    return common::Span<T>{p_data, length};
  }

  /*
   * \param column A JSON object representing a column.
   */
  template <typename T>
  static Columnar<T> ExtractArray(experimental::Json const& column) {
    common::Span<T> s_data { ArrayInterfaceHandler::ExtractData<T>(column) };

    Columnar<T> foreign_col;
    foreign_col.data  = s_data;
    foreign_col.size  = s_data.size();

    common::Span<RBitField8::value_type> s_mask;
    size_t n_bits = ArrayInterfaceHandler::ExtractMask(column, &s_mask);

    foreign_col.valid = RBitField8(s_mask);

    if (s_mask.data()) {
      CHECK_EQ(n_bits, foreign_col.data.size())
          << "Shape of bit mask doesn't match data shape. "
          << "XGBoost doesn't support internal broadcasting.";
    }

    return foreign_col;
  }
};

#define DISPATCH_TYPE(__dispatched_func, __typestr, ...) {              \
    CHECK_EQ(__typestr.size(), 3) << ColumnarErrors::TypestrFormat();   \
    if (__typestr.at(1) == 'f' && __typestr.at(2) == '4') {             \
      __dispatched_func<float>(__VA_ARGS__);                            \
    } else if (__typestr.at(1) == 'f' && __typestr.at(2) == '8') {      \
      __dispatched_func<double>(__VA_ARGS__);                           \
    } else if (__typestr.at(1) == 'i' && __typestr.at(2) == '1') {      \
      __dispatched_func<int8_t>(__VA_ARGS__);                           \
    } else if (__typestr.at(1) == 'i' && __typestr.at(2) == '2') {      \
      __dispatched_func<int16_t>(__VA_ARGS__);                          \
    } else if (__typestr.at(1) == 'i' && __typestr.at(2) == '4') {      \
      __dispatched_func<int32_t>(__VA_ARGS__);                          \
    } else if (__typestr.at(1) == 'i' && __typestr.at(2) == '8') {      \
      __dispatched_func<int64_t>(__VA_ARGS__);                          \
    } else if (__typestr.at(1) == 'u' && __typestr.at(2) == '1') {      \
      __dispatched_func<uint8_t>(__VA_ARGS__);                          \
    } else if (__typestr.at(1) == 'u' && __typestr.at(2) == '2') {      \
      __dispatched_func<uint16_t>(__VA_ARGS__);                         \
    } else if (__typestr.at(1) == 'u' && __typestr.at(2) == '4') {      \
      __dispatched_func<uint32_t>(__VA_ARGS__);                         \
    } else if (__typestr.at(1) == 'u' && __typestr.at(2) == '8') {      \
      __dispatched_func<uint64_t>(__VA_ARGS__);                         \
    } else {                                                            \
      LOG(FATAL) << ColumnarErrors::UnSupportedType(__typestr);         \
    }                                                                   \
  }

}      // namespace xgboost
#endif  // XGBOOST_DATA_COLUMNAR_H_
