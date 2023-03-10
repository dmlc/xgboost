/**
 * Copyright 2023 by XGBoost Contributors
 */
#pragma once
#include <xgboost/context.h>  // for Context

namespace xgboost::ltr {
void TestRankingCache(Context const* ctx);

void TestNDCGCache(Context const* ctx);
}  // namespace xgboost::ltr
