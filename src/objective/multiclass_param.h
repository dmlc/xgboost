/**
 * Copyright 2015-2025, XGBoost Contributors
 *
 * @brief Definition of multi-class classification parameters.
 */
#ifndef XGBOOST_OBJECTIVE_MULTICLASS_PARAM_H_
#define XGBOOST_OBJECTIVE_MULTICLASS_PARAM_H_

#include "xgboost/parameter.h"

namespace xgboost::obj {
struct SoftmaxMultiClassParam : public XGBoostParameter<SoftmaxMultiClassParam> {
  int num_class{1};
  // declare parameters
  DMLC_DECLARE_PARAMETER(SoftmaxMultiClassParam) {
    DMLC_DECLARE_FIELD(num_class).set_lower_bound(1).describe(
        "Number of output class in the multi-class classification.");
  }
};
}  // namespace xgboost::obj
#endif  // XGBOOST_OBJECTIVE_MULTICLASS_PARAM_H_
