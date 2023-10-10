/**
 * Copyright 2023, XGBoost contributors
 */
#include "quantile_loss_utils.h"

#include <cctype>   // for isspace
#include <istream>  // for istream
#include <ostream>  // for ostream
#include <string>   // for string
#include <vector>   // for vector

#include "../common/json_utils.h"  // for TypeCheck
#include "xgboost/json.h"          // for F32Array, get, Number
#include "xgboost/json_io.h"       // for JsonWriter

namespace xgboost::common {
std::ostream& operator<<(std::ostream& os, const ParamFloatArray& array) {
  auto const& t = array.Get();
  xgboost::F32Array arr{t.size()};
  for (std::size_t i = 0; i < t.size(); ++i) {
    arr.Set(i, t[i]);
  }
  std::vector<char> stream;
  xgboost::JsonWriter writer{&stream};
  arr.Save(&writer);
  for (auto c : stream) {
    os << c;
  }
  return os;
}

std::istream& operator>>(std::istream& is, ParamFloatArray& array) {
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

  auto jarr = xgboost::Json::Load(xgboost::StringView{str});
  // return if there's only one element
  if (xgboost::IsA<xgboost::Number>(jarr)) {
    t.emplace_back(xgboost::get<xgboost::Number const>(jarr));
    return is;
  }

  auto jvec = xgboost::get<xgboost::Array const>(jarr);
  for (auto v : jvec) {
    xgboost::TypeCheck<xgboost::Number>(v, "alpha");
    t.emplace_back(get<xgboost::Number const>(v));
  }
  return is;
}

DMLC_REGISTER_PARAMETER(QuantileLossParam);
}  // namespace xgboost::common
