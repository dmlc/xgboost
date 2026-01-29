/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#ifndef TESTS_CPP_OBJECTIVE_TEST_QUANTILE_OBJ_H_
#define TESTS_CPP_OBJECTIVE_TEST_QUANTILE_OBJ_H_

#include <xgboost/context.h>  // for Context

namespace xgboost {

void TestQuantile(Context const* ctx);

void TestQuantileIntercept(Context const* ctx);

}  // namespace xgboost

#endif  // TESTS_CPP_OBJECTIVE_TEST_QUANTILE_OBJ_H_
