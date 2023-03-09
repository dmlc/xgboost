/**
 * Copyright 2023 by XGBoost contributors
 */
#include "ranking_utils.h"

#include <algorithm>          // for copy_n, max, min, none_of, all_of
#include <cstddef>            // for size_t
#include <cstdio>             // for sscanf
#include <exception>          // for exception
#include <functional>         // for greater
#include <iterator>           // for reverse_iterator
#include <string>             // for char_traits, string

#include "algorithm.h"        // for ArgSort
#include "linalg_op.h"        // for cbegin, cend
#include "optional_weight.h"  // for MakeOptionalWeights
#include "threading_utils.h"  // for ParallelFor
#include "xgboost/base.h"     // for bst_group_t
#include "xgboost/context.h"  // for Context
#include "xgboost/data.h"     // for MetaInfo
#include "xgboost/linalg.h"   // for All, TensorView, Range, Tensor, Vector
#include "xgboost/logging.h"  // for Error, LogCheck_EQ, CHECK_EQ

namespace xgboost::ltr {
DMLC_REGISTER_PARAMETER(LambdaRankParam);

std::string ParseMetricName(StringView name, StringView param, position_t* topn, bool* minus) {
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

std::string MakeMetricName(StringView name, position_t topn, bool minus) {
  std::ostringstream ss;
  if (topn == LambdaRankParam::NotSet()) {
    ss << name;
  } else {
    ss << name << "@" << topn;
  }
  if (minus) {
    ss << "-";
  }
  std::string out_name = ss.str();
  return out_name;
}
}  // namespace xgboost::ltr
