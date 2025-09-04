/**
 * Copyright 2023-2025, XGBoost contributors
 */
#include "param_array.h"

#include <cctype>   // for isspace
#include <cstddef>  // for size_t
#include <istream>  // for istream
#include <ostream>  // for ostream
#include <string>   // for string
#include <vector>   // for vector

#include "../common/json_utils.h"  // for TypeCheck
#include "xgboost/json.h"          // for F32Array, get, Number
#include "xgboost/json_io.h"       // for JsonWriter
#include "xgboost/string_view.h"   // for StringView

namespace xgboost::common {

namespace {
std::ostream& WriteStream(std::ostream& os,
                          const ParamArray<float>& array) {  // NOLINT
  auto const& t = array.Get();
  F32Array arr{t.size()};
  for (std::size_t i = 0; i < t.size(); ++i) {
    arr.Set(i, t[i]);
  }
  std::vector<char> stream;
  JsonWriter writer{&stream};
  arr.Save(&writer);
  for (auto c : stream) {
    os << c;
  }
  return os;
}
}  // namespace

std::ostream& operator<<(std::ostream& os, const ParamArray<float>& array) {  // NOLINT
  return WriteStream(os, array);
}

namespace {
std::istream& ReadStream(std::istream& is, ParamArray<float>& array) {  // NOLINT
  auto& t = array.Get();
  t.clear();
  std::string str;
  while (!is.eof()) {
    std::string tmp;
    is >> tmp;
    str += tmp;
  }
  std::size_t head{0};
  // unify notation for parsing.
  while (std::isspace(str[head])) {
    ++head;
  }
  if (str[head] == '(') {
    str[head] = '[';
  }
  auto tail = str.size() - 1;
  while (std::isspace(str[tail])) {
    --tail;
  }
  if (str[tail] == ')') {
    str[tail] = ']';
  }

  auto jarr = Json::Load(StringView{str});
  // return if there's only one element
  if (IsA<Number>(jarr)) {
    t.emplace_back(get<Number const>(jarr));
    return is;
  }
  if (IsA<Integer>(jarr)) {
    t.emplace_back(get<Integer const>(jarr));
    return is;
  }

  auto const& jvec = get<Array const>(jarr);
  for (auto v : jvec) {
    TypeCheck<Number, Integer>(v, array.Name());
    if (IsA<Number>(v)) {
      t.emplace_back(get<Number const>(v));
    } else {
      t.emplace_back(get<Integer const>(v));
    }
  }
  return is;
}
}  // namespace

std::istream& operator>>(std::istream& is, ParamArray<float>& array) {  // NOLINT
  return ReadStream(is, array);
}
}  // namespace xgboost::common
