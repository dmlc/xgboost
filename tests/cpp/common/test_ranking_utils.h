/**
 * Copyright 2023 by XGBoost Contributors
 */
#pragma once
#include <xgboost/context.h>  // for Context

namespace xgboost::ltr {
void TestNDCGCache(Context const* ctx);

void TestMAPCache(Context const* ctx);
}  // namespace xgboost::ltr
