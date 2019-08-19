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
#include "../common/span.h"
#include "../common/bitfield.h"

namespace xgboost {
struct Columnar {
  using mask_type = unsigned char;
  using index_type = int32_t;

  common::Span<float>  data;
  RBitField8 valid;
  int32_t size;
  int32_t null_count;
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
  static char const* toFloat() {
    return "Please convert the input into float32 first.";
  }
  static char const* toUInt() {
    return "Please convert the Group into unsigned 32 bit integers first.";
  }
  static char const* ofType(std::string type) {
    static std::string str;
    str.clear();
    str += " should be of ";
    str += type;
    str += " type.";
    return str.c_str();
  }
};

template <typename PtrType>
PtrType GetPtrFromArrayData(std::map<std::string, Json> const& obj) {
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

}      // namespace xgboost
#endif  // XGBOOST_DATA_COLUMNAR_H_
