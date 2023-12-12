/*!
 * Copyright 2015-2023 by Contributors
 * \file multiclass_param.h
 * \brief Definition of multi-class classification parameters.
 */
#ifndef XGBOOST_OBJECTIVE_MULTICLASS_PARAM_H_
#define XGBOOST_OBJECTIVE_MULTICLASS_PARAM_H_

#include "xgboost/parameter.h"

namespace xgboost {
namespace obj {

struct SoftmaxMultiClassParam : public XGBoostParameter<SoftmaxMultiClassParam> {
  int num_class;
  // declare parameters
  DMLC_DECLARE_PARAMETER(SoftmaxMultiClassParam) {
    DMLC_DECLARE_FIELD(num_class).set_lower_bound(1)
        .describe("Number of output class in the multi-class classification.");
  }
};

}  // namespace obj
}  // namespace xgboost
#endif  // XGBOOST_OBJECTIVE_MULTICLASS_PARAM_H_
