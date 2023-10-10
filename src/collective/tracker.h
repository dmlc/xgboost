/**
 * Copyright 2023, XGBoost Contributors
 */
#pragma once
#include <string>  // for string

#include "xgboost/collective/result.h"  // for Result

namespace xgboost::collective {
// Prob the public IP address of the host, need a better method.
//
// This is directly translated from the previous Python implementation, we should find a
// more riguous approach, can use some expertise in network programming.
[[nodiscard]] Result GetHostAddress(std::string* out);
}  // namespace xgboost::collective
