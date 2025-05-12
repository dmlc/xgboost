/**
 * Copyright 2023-2025, XGBoost Contributors
 */
#ifndef XGBOOST_DATA_BATCH_UTILS_H_
#define XGBOOST_DATA_BATCH_UTILS_H_

#include <cmath>   // for isnan
#include <limits>  // for numeric_limits

#include "xgboost/data.h"  // for BatchParam

namespace xgboost::data::detail {
// At least one batch parameter is initialized.
inline void CheckEmpty(BatchParam const& l, BatchParam const& r) {
  if (!l.Initialized()) {
    CHECK(r.Initialized()) << "Batch parameter is not initialized.";
  }
}

/**
 * \brief Should we regenerate the gradient index?
 *
 * \param old Parameter stored in DMatrix.
 * \param p   New parameter passed in by caller.
 */
inline bool RegenGHist(BatchParam old, BatchParam p) {
  // Parameter is renewed or caller requests a regen
  if (!p.Initialized()) {
    // Empty parameter is passed in, don't regenerate so that we can use gindex in
    // predictor, which doesn't have any training parameter.
    return false;
  }
  return p.regen || old.ParamNotEqual(p);
}

/**
 * @brief Validate the batch parameter from the caller
 */
void CheckParam(BatchParam const& init, BatchParam const& param);

/**
 * @brief Get the default host ratio.
 */
[[nodiscard]] float DftHostRatio(float cache_host_ratio, bool is_validation);

[[nodiscard]] inline bool HostRatioIsAuto(float cache_host_ratio) {
  return std::isnan(cache_host_ratio);
}
}  // namespace xgboost::data::detail

namespace xgboost::cuda_impl {
// Indicator for XGBoost to not concatenate any page.
constexpr std::int64_t MatchingPageBytes() { return 0; }
// Default size of the cached page
constexpr double CachePageRatio() { return 0.125; }
// Indicator for XGBoost to automatically concatenate pages.
constexpr std::int64_t AutoCachePageBytes() { return -1; }
// Use two batch for prefecting. There's always one batch being worked on, while the other
// batch being transferred.
constexpr auto DftPrefetchBatches() { return 2; }
// The ratio of the cache split for external memory. Use -1 to indicate not-set.
constexpr float AutoHostRatio() { return std::numeric_limits<float>::quiet_NaN(); }

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
}  // namespace xgboost::cuda_impl
#endif  // XGBOOST_DATA_BATCH_UTILS_H_
