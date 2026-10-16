#ifndef XGBOOST_METRIC_RANK_METRIC_H_
#define XGBOOST_METRIC_RANK_METRIC_H_
/**
 * Copyright 2023-2026, XGBoost Contributors
 */
#include <memory>  // for shared_ptr

#include "../common/ranking_utils.h"     // for NDCGCache, MAPCache
#include "metric_common.h"               // for PackedReduceResult
#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector

namespace xgboost::metric {
struct AMSEvalKernel {
  using Signature = double(Context const *, HostDeviceVector<float> const &, MetaInfo const &,
                           float);
};

struct CoxEvalKernel {
  using Signature = double(Context const *, HostDeviceVector<float> const &, MetaInfo const &);
};

struct PrecisionEvalKernel {
  using Signature = PackedReduceResult(Context const *, MetaInfo const &,
                                       HostDeviceVector<float> const &,
                                       std::shared_ptr<ltr::PreCache>);
};

struct NDCGEvalKernel {
  using Signature = PackedReduceResult(Context const *, MetaInfo const &,
                                       HostDeviceVector<float> const &, bool,
                                       std::shared_ptr<ltr::NDCGCache>);
};

struct MAPEvalKernel {
  using Signature = PackedReduceResult(Context const *, MetaInfo const &,
                                       HostDeviceVector<float> const &, bool,
                                       std::shared_ptr<ltr::MAPCache>);
};

}  // namespace xgboost::metric
#endif  // XGBOOST_METRIC_RANK_METRIC_H_
