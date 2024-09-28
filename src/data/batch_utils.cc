/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#include "batch_utils.h"

#include "../common/error_msg.h"  // for InconsistentMaxBin

namespace xgboost::data::detail {
void CheckParam(BatchParam const& init, BatchParam const& param) {
  CHECK_EQ(param.max_bin, init.max_bin) << error::InconsistentMaxBin();
  CHECK(!param.regen && param.hess.empty())
      << "Only the `hist` tree method can use the `QuantileDMatrix`.";
}
}  // namespace xgboost::data::detail
