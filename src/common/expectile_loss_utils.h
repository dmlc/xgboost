/**
 * Copyright 2026, XGBoost contributors
 */
#ifndef XGBOOST_COMMON_EXPECTILE_LOSS_UTILS_H_
#define XGBOOST_COMMON_EXPECTILE_LOSS_UTILS_H_

#include <algorithm>  // for all_of

#include "param_array.h"        // for ParamArray
#include "xgboost/logging.h"    // CHECK
#include "xgboost/parameter.h"  // XGBoostParameter

namespace xgboost::common {
struct ExpectileLossParam : public XGBoostParameter<ExpectileLossParam> {
  ParamArray<float> expectile_alpha{"expectile_alpha"};
  DMLC_DECLARE_PARAMETER(ExpectileLossParam) {
    DMLC_DECLARE_FIELD(expectile_alpha)
        .describe("List of expectiles for expectile loss.")
        .set_default(ParamArray<float>{"expectile_alpha"});
  }
  void Validate() const {
    CHECK(GetInitialised());
    CHECK(!expectile_alpha.Get().empty());
    auto const& array = expectile_alpha.Get();
    auto valid =
        std::all_of(array.cbegin(), array.cend(), [](auto q) { return q >= 0.0 && q <= 1.0; });
    CHECK(valid) << "expectile alpha must be in the range [0.0, 1.0].";
  }
};
}  // namespace xgboost::common
#endif  // XGBOOST_COMMON_EXPECTILE_LOSS_UTILS_H_
