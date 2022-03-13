#ifndef XGBOOST_COMMON_REGULARIZED_H_
#define XGBOOST_COMMON_REGULARIZED_H_
/*!
 * Copyright 2022 by XGBoost Contributors
 */
#include "xgboost/parameter.h"

namespace xgboost {
struct BinaryRegularizationParam : public XGBoostParameter<BinaryRegularizationParam> {
  float fairness{0.0f};
  DMLC_DECLARE_PARAMETER(BinaryRegularizationParam) {
    DMLC_DECLARE_FIELD(fairness)
        .set_range(0.0f, 1.0f)
        .describe("The strength of the regularizer for fairness XGBoost.");
  }
};
}  // namespace xgboost
#endif  // XGBOOST_COMMON_REGULARIZED_H_
