/**
 * Copyright 2026, XGBoost Contributors
 * \file normal_obj.h
 * \brief Shared declarations for normal distribution regression.
 */
#ifndef XGBOOST_OBJECTIVE_NORMAL_OBJ_H_
#define XGBOOST_OBJECTIVE_NORMAL_OBJ_H_

#include <algorithm>  // for clamp, max
#include <cmath>      // for exp, log, sqrt, abs
#include <limits>     // for numeric_limits

#include "xgboost/base.h"                // for GradientPair
#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/linalg.h"              // for Matrix, Vector

namespace xgboost::obj {
constexpr float kNormalMinVariance = std::numeric_limits<float>::epsilon();

struct NormalGradient {
  XGBOOST_DEVICE static GradientPair FinitePair(double grad, double hess) {
    // Leave headroom for squaring gradients in the tree gain calculation. Scale the pair
    // together so numerical saturation preserves its Newton step (but changes its weight
    // relative to other rows). Ordinary, representable pairs are unchanged.
    auto limit = std::sqrt(static_cast<double>(std::numeric_limits<float>::max())) / 2.0;
    auto magnitude = std::max(std::abs(grad), hess);
    auto scale = magnitude > limit ? limit / magnitude : 1.0;
    return {static_cast<float>(grad * scale), static_cast<float>(hess * scale)};
  }

  XGBOOST_DEVICE void operator()(float mean, float log_variance, float label, float weight,
                                 GradientPair* out_mean, GradientPair* out_log_variance) const {
    // A custom intercept or base margin can start far below the noise-floor equilibrium.
    // Bound the exponential before evaluating it, and use double intermediates so finite
    // float inputs cannot overflow while forming weighted squared residuals.
    auto residual = static_cast<double>(mean) - label;
    auto log_limit = std::log(static_cast<double>(std::numeric_limits<float>::max()));
    auto precision =
        std::exp(std::clamp(-static_cast<double>(log_variance), -log_limit, log_limit));
    // A fixed noise floor keeps the scale optimum finite when the mean interpolates the label.
    // Without it, zero residuals drive log variance toward -infinity until expf overflows.
    auto standardized_residual = (residual * residual + kNormalMinVariance) * precision;

    *out_mean = FinitePair(weight * residual * precision, weight * precision);
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
    *out_log_variance = FinitePair(weight * grad_log_variance, weight * hess_log_variance);
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
