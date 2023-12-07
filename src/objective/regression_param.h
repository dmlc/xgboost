/*!
 * Copyright 2015-2023 by Contributors
 * \file multiclass_param.h
 * \brief Definition of single-value regression and classification parameters.
 */
#ifndef XGBOOST_OBJECTIVE_REGRESSION_PARAM_H_
#define XGBOOST_OBJECTIVE_REGRESSION_PARAM_H_

#include "xgboost/parameter.h"

namespace xgboost {
namespace obj {

struct RegLossParam : public XGBoostParameter<RegLossParam> {
  float scale_pos_weight;
  // declare parameters
  DMLC_DECLARE_PARAMETER(RegLossParam) {
    DMLC_DECLARE_FIELD(scale_pos_weight).set_default(1.0f).set_lower_bound(0.0f)
      .describe("Scale the weight of positive examples by this factor");
  }
};

}  // namespace obj
}  // namespace xgboost
#endif  // XGBOOST_OBJECTIVE_REGRESSION_PARAM_H_
