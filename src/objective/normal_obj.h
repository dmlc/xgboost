/**
 * Copyright 2026, XGBoost Contributors
 * \file normal_obj.h
 * \brief Shared declarations for normal distribution regression.
 */
#ifndef XGBOOST_OBJECTIVE_NORMAL_OBJ_H_
#define XGBOOST_OBJECTIVE_NORMAL_OBJ_H_

#include <cmath>   // for expf
#include <limits>  // for numeric_limits

#include "xgboost/base.h"                // for GradientPair
#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/linalg.h"              // for Matrix, Vector

namespace xgboost::obj {
constexpr float kNormalMinVariance = std::numeric_limits<float>::epsilon();

struct NormalGradient {
  XGBOOST_DEVICE void operator()(float mean, float log_variance, float label, float weight,
                                 GradientPair* out_mean, GradientPair* out_log_variance) const {
    auto residual = mean - label;
    auto precision = expf(-log_variance);
    // A fixed noise floor keeps the scale optimum finite when the mean interpolates the label.
    // Without it, zero residuals drive log variance toward -infinity until expf overflows.
    auto standardized_residual = (residual * residual + kNormalMinVariance) * precision;

    *out_mean = {weight * residual * precision, weight * precision};
    auto grad_log_variance = 0.5f * (1.0f - standardized_residual);
    // Holding the mean fixed, let R be the leaf-average standardized squared residual. The
    // exact log-variance update is log(R), while the observed Newton update 1 - 1/R is
    // unbounded as R tends to zero. This curvature gives
    //
    //   d = 3 * (R - 1) / (1 + 2 * R),
    //
    // which is bounded to (-3, 3/2), descends the fixed-mean loss, and gives a conservative
    // quadratic gain estimate. It is the sign-reversed Poisson bounded-step construction.
    auto hess_log_variance = (1.0f + 2.0f * standardized_residual) / 6.0f;
    *out_log_variance = {weight * grad_log_variance, weight * hess_log_variance};
  }
};

struct NormalGradientKernel {
  using Signature = void(Context const*, HostDeviceVector<float> const&, MetaInfo const&,
                         linalg::Matrix<GradientPair>*);
};

struct NormalInitEstimationKernel {
  using Signature = void(Context const*, MetaInfo const&, linalg::Vector<float>*);
};
}  // namespace xgboost::obj

#endif  // XGBOOST_OBJECTIVE_NORMAL_OBJ_H_
