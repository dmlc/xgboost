/**
 * Copyright 2026, XGBoost Contributors
 * \file normal_metric.h
 * \brief Shared declarations for the normal negative log-likelihood metric.
 */
#ifndef XGBOOST_METRIC_NORMAL_METRIC_H_
#define XGBOOST_METRIC_NORMAL_METRIC_H_

#include <cmath>  // for expf, fabsf, logf

#include "metric_common.h"               // for PackedReduceResult
#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector

namespace xgboost::metric {
struct EvalNormalNLogLik {
  XGBOOST_DEVICE float operator()(float label, float mean, float log_variance) const {
    constexpr float kLogTwoPi = 1.8378770664093453f;
    auto residual = label - mean;
    auto standardized_residual =
        residual == 0.0f ? 0.0f : expf(2.0f * logf(fabsf(residual)) - log_variance);
    return 0.5f * (kLogTwoPi + log_variance + standardized_residual);
  }
};

struct NormalEvalKernel {
  using Signature = PackedReduceResult(Context const*, HostDeviceVector<float> const&,
                                       MetaInfo const&);
};
}  // namespace xgboost::metric

#endif  // XGBOOST_METRIC_NORMAL_METRIC_H_
