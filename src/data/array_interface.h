/*!
 * Copyright 2019 by Contributors
 * \file array_interface.h
 * \brief Basic structure holding a reference to arrow columnar data format.
 */
#ifndef XGBOOST_DATA_ARRAY_INTERFACE_H_
#define XGBOOST_DATA_ARRAY_INTERFACE_H_

#include <cinttypes>
#include <map>
#include <string>
#include <utility>

#include "xgboost/data.h"
#include "xgboost/json.h"
#include "xgboost/logging.h"
#include "xgboost/span.h"
#include "../common/bitfield.h"

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
    return "Only version 1 of `__cuda_array_interface__' is supported.";
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

  static std::string UnSupportedType(const char (&typestr)[3]) {
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
    if (array.find("version") == array.cend()) {
      LOG(FATAL) << "Missing `version' field for array interface";
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

  static std::pair<size_t, size_t> ExtractShape(
      std::map<std::string, Json> const& column) {
    auto j_shape = get<Array const>(column.at("shape"));
    auto typestr = get<String const>(column.at("typestr"));
    if (column.find("strides") != column.cend()) {
      if (!IsA<Null>(column.at("strides"))) {
        auto strides = get<Array const>(column.at("strides"));
        CHECK_EQ(strides.size(), j_shape.size())
            << ArrayInterfaceErrors::Dimension(1);
        CHECK_EQ(get<Integer>(strides.at(0)), typestr.at(2) - '0')
            << ArrayInterfaceErrors::Contigious();
      }
    }

    if (j_shape.size() == 1) {
      return {static_cast<size_t>(get<Integer const>(j_shape.at(0))), 1};
    } else {
      CHECK_EQ(j_shape.size(), 2)
          << "Only 1D or 2-D arrays currently supported.";
      return {static_cast<size_t>(get<Integer const>(j_shape.at(0))),
              static_cast<size_t>(get<Integer const>(j_shape.at(1)))};
    }
  }
  template <typename T>
  static common::Span<T> ExtractData(std::map<std::string, Json> const& column) {
    Validate(column);

    auto typestr = get<String const>(column.at("typestr"));
    CHECK_EQ(typestr.at(1),   TypeChar<T>())
        << "Input data type and typestr mismatch. typestr: " << typestr;
    CHECK_EQ(typestr.at(2),   static_cast<char>(sizeof(T) + 48))
        << "Input data type and typestr mismatch. typestr: " << typestr;


    auto shape = ExtractShape(column);

    T* p_data = ArrayInterfaceHandler::GetPtrFromArrayData<T*>(column);
    return common::Span<T>{p_data, shape.first * shape.second};
  }
};

// A view over __array_interface__
class ArrayInterface {
 public:
  ArrayInterface() = default;
  explicit ArrayInterface(std::map<std::string, Json> const &column,
                          bool allow_mask = true) {
    ArrayInterfaceHandler::Validate(column);
    data = ArrayInterfaceHandler::GetPtrFromArrayData<void*>(column);
    CHECK(data) << "Column is null";
    auto shape = ArrayInterfaceHandler::ExtractShape(column);
    num_rows = shape.first;
    num_cols = shape.second;

    if (allow_mask) {
      common::Span<RBitField8::value_type> s_mask;
      size_t n_bits = ArrayInterfaceHandler::ExtractMask(column, &s_mask);

      valid = RBitField8(s_mask);

      if (s_mask.data()) {
        CHECK_EQ(n_bits, num_rows)
            << "Shape of bit mask doesn't match data shape. "
            << "XGBoost doesn't support internal broadcasting.";
      }
    } else {
      CHECK(column.find("mask") == column.cend())
          << "Masked array is not yet supported.";
    }

    auto typestr = get<String const>(column.at("typestr"));
    type[0] = typestr.at(0);
    type[1] = typestr.at(1);
    type[2] = typestr.at(2);
    this->CheckType();
  }

  void CheckType() const {
    if (type[1] == 'f' && type[2] == '4') {
      return;
    } else if (type[1] == 'f' && type[2] == '8') {
      return;
    } else if (type[1] == 'i' && type[2] == '1') {
      return;
    } else if (type[1] == 'i' && type[2] == '2') {
      return;
    } else if (type[1] == 'i' && type[2] == '4') {
      return;
    } else if (type[1] == 'i' && type[2] == '8') {
      return;
    } else if (type[1] == 'u' && type[2] == '1') {
      return;
    } else if (type[1] == 'u' && type[2] == '2') {
      return;
    } else if (type[1] == 'u' && type[2] == '4') {
      return;
    } else if (type[1] == 'u' && type[2] == '8') {
      return;
    } else {
      LOG(FATAL) << ArrayInterfaceErrors::UnSupportedType(type);
      return;
    }
  }

  XGBOOST_DEVICE float GetElement(size_t idx) const {
    if (type[1] == 'f' && type[2] == '4') {
      return reinterpret_cast<float*>(data)[idx];
    } else if (type[1] == 'f' && type[2] == '8') {
      return reinterpret_cast<double*>(data)[idx];
    } else if (type[1] == 'i' && type[2] == '1') {
      return reinterpret_cast<int8_t*>(data)[idx];
    } else if (type[1] == 'i' && type[2] == '2') {
      return reinterpret_cast<int16_t*>(data)[idx];
    } else if (type[1] == 'i' && type[2] == '4') {
      return reinterpret_cast<int32_t*>(data)[idx];
    } else if (type[1] == 'i' && type[2] == '8') {
      return reinterpret_cast<int64_t*>(data)[idx];
    } else if (type[1] == 'u' && type[2] == '1') {
      return reinterpret_cast<uint8_t*>(data)[idx];
    } else if (type[1] == 'u' && type[2] == '2') {
      return reinterpret_cast<uint16_t*>(data)[idx];
    } else if (type[1] == 'u' && type[2] == '4') {
      return reinterpret_cast<uint32_t*>(data)[idx];
    } else if (type[1] == 'u' && type[2] == '8') {
      return reinterpret_cast<uint64_t*>(data)[idx];
    } else {
      SPAN_CHECK(false);
      return 0;
    }
  }

  RBitField8 valid;
  int32_t num_rows;
  int32_t num_cols;
  void* data;
  char type[3];
};

}  // namespace xgboost
#endif  // XGBOOST_DATA_ARRAY_INTERFACE_H_
