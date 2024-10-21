/**
 * Copyright 2020-2023 by XGBoost Contributors
 */
#ifndef XGBOOST_TEST_REGRESSION_OBJ_H_
#define XGBOOST_TEST_REGRESSION_OBJ_H_

#include <xgboost/context.h>  // for Context

namespace xgboost {

void TestLinearRegressionGPair(const Context* ctx);

void TestSquaredLog(const Context* ctx);

void TestLogisticRegressionGPair(const Context* ctx);

void TestLogisticRegressionBasic(const Context* ctx);

void TestsLogisticRawGPair(const Context* ctx);

}  // namespace xgboost

#endif  // XGBOOST_TEST_REGRESSION_OBJ_H_