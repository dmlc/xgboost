#ifndef XGBOOST_METRIC_RANK_METRIC_H_
#define XGBOOST_METRIC_RANK_METRIC_H_
/**
 * Copyright 2023 by XGBoost Contributors
 */
#include <memory>  // for shared_ptr

#include "../common/common.h"            // for AssertGPUSupport
#include "../common/ranking_utils.h"     // for NDCGCache, MAPCache
#include "metric_common.h"               // for PackedReduceResult
#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector

namespace xgboost::metric::cuda_impl {
PackedReduceResult NDCGScore(Context const *ctx, MetaInfo const &info,
                             HostDeviceVector<float> const &predt, bool minus,
                             std::shared_ptr<ltr::NDCGCache> p_cache);

PackedReduceResult MAPScore(Context const *ctx, MetaInfo const &info,
                            HostDeviceVector<float> const &predt, bool minus,
                            std::shared_ptr<ltr::MAPCache> p_cache);

PackedReduceResult PreScore(Context const *ctx, MetaInfo const &info,
                            HostDeviceVector<float> const &predt,
                            std::shared_ptr<ltr::PreCache> p_cache);

#if !defined(XGBOOST_USE_CUDA)
inline PackedReduceResult NDCGScore(Context const *, MetaInfo const &,
                                    HostDeviceVector<float> const &, bool,
                                    std::shared_ptr<ltr::NDCGCache>) {
  common::AssertGPUSupport();
  return {};
}

inline PackedReduceResult MAPScore(Context const *, MetaInfo const &,
                                   HostDeviceVector<float> const &, bool,
                                   std::shared_ptr<ltr::MAPCache>) {
  common::AssertGPUSupport();
  return {};
}

inline PackedReduceResult PreScore(Context const *, MetaInfo const &,
                                   HostDeviceVector<float> const &,
                                   std::shared_ptr<ltr::PreCache>) {
  common::AssertGPUSupport();
  return {};
}
#endif
}  // namespace xgboost::metric::cuda_impl
#endif  // XGBOOST_METRIC_RANK_METRIC_H_
