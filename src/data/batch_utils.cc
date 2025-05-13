/**
 * Copyright 2023-2025, XGBoost Contributors
 */
#include "batch_utils.h"

#include "../common/error_msg.h"  // for InconsistentMaxBin

namespace xgboost::data::detail {
void CheckParam(BatchParam const& init, BatchParam const& param) {
  CHECK_EQ(param.max_bin, init.max_bin) << error::InconsistentMaxBin();
  CHECK(!param.regen && param.hess.empty())
      << "Only the `hist` tree method can use the `QuantileDMatrix`.";
}

[[nodiscard]] float DftHostRatio(float cache_host_ratio, bool is_validation) {
  if (is_validation) {
    // Don't split the cache if this is a validation dataset.
    return 1.0;
  }
  if (HostRatioIsAuto(cache_host_ratio)) {
    // Only NVML has the API to detect the topology. We will leave it as-is for now.
    cache_host_ratio = 1.0;
    return cache_host_ratio;
  }
  // Use user config.
  CHECK_GE(cache_host_ratio, 0.0f) << error::CacheHostRatioInvalid();
  CHECK_LE(cache_host_ratio, 1.0f) << error::CacheHostRatioInvalid();
  return cache_host_ratio;
}
}  // namespace xgboost::data::detail
