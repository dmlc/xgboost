/**
 * Copyright 2020-2024 by XGBoost Contributors
 */
#ifndef XGBOOST_TEST_QUANTILE_OBJ_H_
#define XGBOOST_TEST_QUANTILE_OBJ_H_

#include <xgboost/context.h>  // for Context

namespace xgboost {

void TestQuantile(const Context* ctx);

void TestQuantileIntercept(const Context* ctx);

}  // namespace xgboost

#endif  // XGBOOST_TEST_REGRESSION_OBJ_H_
