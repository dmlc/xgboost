/**
 * Copyright 2023-2025, XGBoost contributors
 */
#ifndef XGBOOST_COMMON_QUANTILE_LOSS_UTILS_H_
#define XGBOOST_COMMON_QUANTILE_LOSS_UTILS_H_

#include <algorithm>  // for all_of

#include "param_array.h"        // for ParamArray
#include "xgboost/logging.h"    // CHECK
#include "xgboost/parameter.h"  // XGBoostParameter

namespace xgboost::common {
struct QuantileLossParam : public XGBoostParameter<QuantileLossParam> {
  ParamArray<float> quantile_alpha{"quantile_alpha"};
  DMLC_DECLARE_PARAMETER(QuantileLossParam) {
    DMLC_DECLARE_FIELD(quantile_alpha)
        .describe("List of quantiles for quantile loss.")
        .set_default(ParamArray<float>{"quantile_alpha"});
  }
  void Validate() const {
    CHECK(GetInitialised());
    CHECK(!quantile_alpha.Get().empty());
    auto const& array = quantile_alpha.Get();
    auto valid =
        std::all_of(array.cbegin(), array.cend(), [](auto q) { return q >= 0.0 && q <= 1.0; });
    CHECK(valid) << "quantile alpha must be in the range [0.0, 1.0].";
  }
};
}  // namespace xgboost::common
#endif  // XGBOOST_COMMON_QUANTILE_LOSS_UTILS_H_
