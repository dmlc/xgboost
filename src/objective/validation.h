/**
 * Copyright 2022 by XGBoost Contributors
 */
#include <xgboost/data.h>  //MetaInfo

namespace xgboost {
namespace obj {
inline void CheckInitInputs(MetaInfo const& info) {
  CHECK_EQ(info.labels.Shape(0), info.num_row_) << "Invalid shape of labels.";
  if (!info.weights_.Empty()) {
    CHECK_EQ(info.weights_.Size(), info.num_row_)
        << "Number of weights should be equal to number of data points.";
  }
}
}  // namespace obj
}  // namespace xgboost
