/**
 * Copyright 2020-2023 by XGBoost Contributors
 */
#ifndef XGBOOST_TEST_MULTICLASS_OBJ_H_
#define XGBOOST_TEST_MULTICLASS_OBJ_H_

#include <xgboost/context.h>  // for Context

namespace xgboost {

void TestSoftmaxMultiClassObjGPair(const Context* ctx);

void TestSoftmaxMultiClassBasic(const Context* ctx);

void TestSoftprobMultiClassBasic(const Context* ctx);

}  // namespace xgboost

#endif  // XGBOOST_TEST_MULTICLASS_OBJ_H_
