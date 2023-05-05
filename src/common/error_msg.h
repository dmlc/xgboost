/**
 * Copyright 2023 by XGBoost contributors
 *
 * \brief Common error message for various checks.
 */
#ifndef XGBOOST_COMMON_ERROR_MSG_H_
#define XGBOOST_COMMON_ERROR_MSG_H_

#include <cinttypes>  // for uint64_t
#include <limits>     // for numeric_limits

#include "xgboost/base.h"  // for bst_feature_t
#include "xgboost/logging.h"
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

constexpr StringView NoF128() {
  return "128-bit floating point is not supported on current platform.";
}

constexpr StringView InconsistentMaxBin() {
  return "Inconsistent `max_bin`. `max_bin` should be the same across different QuantileDMatrix, "
         "and consistent with the Booster being trained.";
}

constexpr StringView UnknownDevice() { return "Unknown device type."; }

inline void MaxFeatureSize(std::uint64_t n_features) {
  auto max_n_features = std::numeric_limits<bst_feature_t>::max();
  CHECK_LE(n_features, max_n_features)
      << "Unfortunately, XGBoost does not support data matrices with "
      << std::numeric_limits<bst_feature_t>::max() << " features or greater";
}
}  // namespace xgboost::error
#endif  // XGBOOST_COMMON_ERROR_MSG_H_
