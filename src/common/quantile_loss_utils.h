/**
 * Copyright 2023 by XGBoost contributors
 */
#ifndef XGBOOST_COMMON_QUANTILE_LOSS_UTILS_H_
#define XGBOOST_COMMON_QUANTILE_LOSS_UTILS_H_

#include <algorithm>            // std::all_of
#include <istream>              // std::istream
#include <ostream>              // std::ostream
#include <vector>               // std::vector

#include "xgboost/logging.h"    // CHECK
#include "xgboost/parameter.h"  // XGBoostParameter

namespace xgboost {
namespace common {
// A shim to enable ADL for parameter parsing. Alternatively, we can put the stream
// operators in std namespace, which seems to be less ideal.
class ParamFloatArray {
  std::vector<float> values_;

 public:
  std::vector<float>& Get() { return values_; }
  std::vector<float> const& Get() const { return values_; }
  decltype(values_)::const_reference operator[](decltype(values_)::size_type i) const {
    return values_[i];
  }
};

// For parsing quantile parameters. Input can be a string to a single float or a list of
// floats.
std::ostream& operator<<(std::ostream& os, const ParamFloatArray& t);
std::istream& operator>>(std::istream& is, ParamFloatArray& t);

struct QuantileLossParam : public XGBoostParameter<QuantileLossParam> {
  ParamFloatArray quantile_alpha;
  DMLC_DECLARE_PARAMETER(QuantileLossParam) {
    DMLC_DECLARE_FIELD(quantile_alpha).describe("List of quantiles for quantile loss.");
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
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_QUANTILE_LOSS_UTILS_H_
