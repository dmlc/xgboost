/**
 * Copyright 2023 by XGBoost contributors
 *
 * \brief Common error message for various checks.
 */
#ifndef XGBOOST_COMMON_ERROR_MSG_H_
#define XGBOOST_COMMON_ERROR_MSG_H_

#include "xgboost/string_view.h"  // for StringView

namespace xgboost::error {
constexpr StringView GroupWeight() {
  return "Size of weight must equal to the number of query groups when ranking group is used.";
}

constexpr StringView GroupSize() {
  return "Invalid query group structure. The number of rows obtained from group doesn't equal to ";
}

constexpr StringView LabelScoreSize() {
  return "The size of label doesn't match the size of prediction.";
}

constexpr StringView InfInData() {
  return "Input data contains `inf` or a value too large, while `missing` is not set to `inf`";
}
}  // namespace xgboost::error
#endif  // XGBOOST_COMMON_ERROR_MSG_H_
