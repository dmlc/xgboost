/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#ifndef XGBOOST_TEST_QUANTILE_OBJ_H_
#define XGBOOST_TEST_QUANTILE_OBJ_H_

#include <xgboost/context.h>  // for Context

namespace xgboost {

void TestQuantile(Context const* ctx);

void TestQuantileIntercept(Context const* ctx);

}  // namespace xgboost

#endif  // XGBOOST_TEST_REGRESSION_OBJ_H_
