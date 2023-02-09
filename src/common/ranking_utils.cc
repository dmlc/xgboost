/**
 * Copyright 2023 by XGBoost contributors
 */
#include "ranking_utils.h"

#include <cstdint>                // std::uint32_t
#include <sstream>                // std::ostringstream
#include <string>                 // std::string,std::sscanf

#include "xgboost/string_view.h"  // StringView

namespace xgboost {
namespace ltr {
std::string MakeMetricName(StringView name, StringView param, std::uint32_t* topn, bool* minus) {
  std::string out_name;
  if (!param.empty()) {
    std::ostringstream os;
    if (std::sscanf(param.c_str(), "%u[-]?", topn) == 1) {
      os << name << '@' << param;
      out_name = os.str();
    } else {
      os << name << param;
      out_name = os.str();
    }
    if (*param.crbegin() == '-') {
      *minus = true;
    }
  } else {
    out_name = name.c_str();
  }
  return out_name;
}
}  // namespace ltr
}  // namespace xgboost
