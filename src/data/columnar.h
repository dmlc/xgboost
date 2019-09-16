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
#include "xgboost/json.h"
#include "xgboost/logging.h"
#include "../common/span.h"
#include "../common/bitfield.h"

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
    return "`typestr` should be of format <endian><type><size>.";
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
    return "Only version 1 of __cuda_array_interface__ is being supported.";
  }
  static char const* ofType(std::string const& type) {
    static std::string str;
    str.clear();
    str += " should be of ";
    str += type;
    str += " type.";
    return str.c_str();
  }
  static std::string UnknownTypeStr(std::string const& typestr) {
    return "typestr from array interface: " + typestr + " is not supported.";
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
      default:
        LOG(FATAL) << "Invalid type code: " << c << " in typestr of input array interface.";
        return "";
    }
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
      LOG(FATAL) << "Missing version field for array interface";
    }
    auto version = get<Integer const>(array.at("version"));
    CHECK_EQ(version, 1) << ColumnarErrors::Version();

    if (array.find("typestr") == array.cend()) {
      LOG(FATAL) << "Missing typestr field for array interface";
    }
    auto typestr = get<String const>(array.at("typestr"));
    CHECK_EQ(typestr.size(),    3) << ColumnarErrors::TypestrFormat();
    CHECK_NE(typestr.front(), '>') << ColumnarErrors::BigEndian();

    if (array.find("shape") == array.cend()) {
      LOG(FATAL) << "Missing shape field for array interface";
    }
    if (array.find("data") == array.cend()) {
      LOG(FATAL) << "Missing data field for array interface";
    }
  }

  // Find null mask (validity mask) field
  // Mask object is also an array interface, but with different requirements.
  static void ExtractMask(std::map<std::string, Json> const& column,
                             common::Span<RBitField8::value_type>* p_out) {
    auto& s_mask = *p_out;
    if (column.find("mask") != column.cend()) {
      auto const& j_mask = get<Object const>(column.at("mask"));
      Validate(j_mask);

      auto p_mask = GetPtrFromArrayData<RBitField8::value_type*>(j_mask);

      auto j_shape = get<Array const>(j_mask.at("shape"));
      CHECK_EQ(j_shape.size(), 1) << ColumnarErrors::Dimension(1);
      CHECK_EQ(get<Integer>(j_shape.front()) % 8, 0) <<
          "Length of validity mask must be a multiple of 8 bytes.";
      int64_t size = get<Integer>(j_shape.at(0)) *
                     sizeof(unsigned char) / sizeof(RBitField8::value_type);
      auto typestr = get<String const>(j_mask.at("typestr"));

      if (typestr.at(1) == 't') {
        CHECK_EQ(typestr.at(2),   '1') << "There can be only 1 bit in each entry of bitfield.";
      } else if (typestr.at(1) == 'i') {
        CHECK_EQ(typestr.at(2),   '1') << "mask with integer type should be of 1 byte per integer.";
      } else {
        LOG(FATAL) << "mask must be of integer type or bit field type.";
      }

      // For now this is just 1
      int64_t const type_length = typestr.at(2) - 48;
      s_mask = {p_mask, size / type_length};
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

    auto j_shape = get<Array const>(column.at("shape"));
    CHECK_EQ(j_shape.size(), 1) << ColumnarErrors::Dimension(1);

    if (column.find("strides") != column.cend()) {
      auto strides = get<Array const>(column.at("strides"));
      CHECK_EQ(strides.size(), 1)              << ColumnarErrors::Dimension(1);
      CHECK_EQ(get<Integer>(strides.at(0)), 4) << ColumnarErrors::Contigious();
    }

    auto length = get<Integer const>(j_shape.at(0));

    T* p_data = ArrayInterfaceHandler::GetPtrFromArrayData<T*>(column);
    return common::Span<T>{p_data, length};
  }

  template <typename T>
  static Columnar<T> ExtractArray(std::map<std::string, Json> const& column) {
    common::Span<T> s_data { ArrayInterfaceHandler::ExtractData<T>(column) };

    Columnar<T> foreign_col;
    foreign_col.data  = s_data;
    foreign_col.size  = s_data.size();

    common::Span<RBitField8::value_type> s_mask;
    ArrayInterfaceHandler::ExtractMask(column, &s_mask);

    foreign_col.valid = RBitField8(s_mask);

    return foreign_col;
  }
};

#define DISPATCH_TYPE(__dispatched_func, __typestr, ...) {              \
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
      LOG(FATAL) << ColumnarErrors::UnknownTypeStr(__typestr);          \
    }                                                                   \
  }

}      // namespace xgboost
#endif  // XGBOOST_DATA_COLUMNAR_H_
