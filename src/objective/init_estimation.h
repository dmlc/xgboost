/**
 * Copyright 2022-2023 by XGBoost contributors
 */
#ifndef XGBOOST_OBJECTIVE_INIT_ESTIMATION_H_
#define XGBOOST_OBJECTIVE_INIT_ESTIMATION_H_
#include "xgboost/data.h"       // MetaInfo
#include "xgboost/linalg.h"     // Tensor
#include "xgboost/objective.h"  // ObjFunction

namespace xgboost::obj {
class FitIntercept : public ObjFunction {
  void InitEstimation(MetaInfo const& info, linalg::Vector<float>* base_score) const override;
};

inline void CheckInitInputs(MetaInfo const& info) {
  CHECK_EQ(info.labels.Shape(0), info.num_row_) << "Invalid shape of labels.";
  if (!info.weights_.Empty()) {
    CHECK_EQ(info.weights_.Size(), info.num_row_)
        << "Number of weights should be equal to number of data points.";
  }
}
}  // namespace xgboost::obj
#endif  // XGBOOST_OBJECTIVE_INIT_ESTIMATION_H_
