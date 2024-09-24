/**
 * Copyright 2024, XGBoost Contributors
 */
#pragma once

#include "xgboost/data.h"  // for BatchParam

namespace xgboost::data::cuda_impl {
// Use two batch for prefecting. There's always one batch being worked on, while the other
// batch being transferred.
constexpr auto DftPrefetchBatches() { return 2; }

// Empty parameter to prevent regen, only used to control external memory prefetching.
//
// Both the approx and hist initializes the DMatrix before creating the actual
// implementation (InitDataOnce). Therefore, the `GPUHistMakerDevice` can use an empty
// parameter to avoid any regen.
inline BatchParam StaticBatch(bool prefetch_copy) {
  BatchParam p;
  p.prefetch_copy = prefetch_copy;
  p.n_prefetch_batches = DftPrefetchBatches();
  return p;
}
}  // namespace xgboost::data::cuda_impl
