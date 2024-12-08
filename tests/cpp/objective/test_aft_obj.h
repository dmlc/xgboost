/**
 * Copyright 2020-2024 by XGBoost Contributors
 */
#ifndef XGBOOST_TEST_AFT_OBJ_H_
#define XGBOOST_TEST_AFT_OBJ_H_

#include <xgboost/context.h>  // for Context

namespace xgboost::common {

void TestAFTObjConfiguration(const Context* ctx);

void TestAFTObjGPairUncensoredLabels(const Context* ctx);

void TestAFTObjGPairLeftCensoredLabels(const Context* ctx);

void TestAFTObjGPairRightCensoredLabels(const Context* ctx);

void TestAFTObjGPairIntervalCensoredLabels(const Context* ctx);

}  // namespace xgboost::common

#endif  // XGBOOST_TEST_AFT_OBJ_H_
