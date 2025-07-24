#ifndef XGBOOST_COMMON_PSEUDO_HUBER_H_
#define XGBOOST_COMMON_PSEUDO_HUBER_H_
/*!
 * Copyright 2022, by XGBoost Contributors
 */
#include "xgboost/parameter.h"

namespace xgboost {
struct PseudoHuberParam : public XGBoostParameter<PseudoHuberParam> {
  float huber_slope{1.0};

  DMLC_DECLARE_PARAMETER(PseudoHuberParam) {
    DMLC_DECLARE_FIELD(huber_slope)
        .set_default(1.0f)
        .describe("The delta term in Pseudo-Huber loss.");
  }
};
}  // namespace xgboost
#endif  // XGBOOST_COMMON_PSEUDO_HUBER_H_
