/**
 * Copyright 2014-2026, XGBoost Contributors
 *
 * @brief The regularised objective of a tree leaf, shared by all tree methods and devices.
 *
 * A leaf with gradient sum @f$ G @f$, Hessian sum @f$ H @f$ and weight @f$ w @f$ has the
 * second-order objective
 *
 *   @f$ obj(w) = G w + \frac{1}{2} (H + \lambda) w^2 + \alpha |w| @f$.
 *
 * Gains are reported as @f$ -2 \, obj(w) @f$. The optimal weight is
 * @f$ w^* = -T(G) / (H + \lambda) @f$, with @f$ T @f$ the soft threshold of ThresholdL1, and its
 * gain is @f$ T(G)^2 / (H + \lambda) @f$. `max_delta_step` and monotone constraints clamp the
 * weight, after which the gain is evaluated at the clamped weight.
 *
 * Every weight and gain formula belongs in this header, so that the CPU, CUDA and SYCL
 * implementations cannot drift apart.
 */
#ifndef XGBOOST_TREE_SPLIT_GAIN_H_
#define XGBOOST_TREE_SPLIT_GAIN_H_

#include <cmath>        // for abs
#include <type_traits>  // for enable_if_t, is_floating_point_v

#include "../common/math.h"  // for Sqr
#include "xgboost/base.h"    // for XGBOOST_DEVICE

namespace xgboost::tree {
/**
 * @brief Whether both children satisfy the Hessian requirement for a split.
 *
 * Vector-leaf callers pass the normalized Hessian trace for each child.
 */
template <typename TrainingParams, typename T>
XGBOOST_DEVICE bool IsValidSplit(TrainingParams const &p, T left_hess, T right_hess) {
  return left_hess > 0.0 && right_hess > 0.0 && left_hess >= p.min_child_weight &&
         right_hess >= p.min_child_weight;
}

/**
 * @brief Function for L1 cost
 *
 *  @f$ L1(G, a) = sign(G) max(|G| - a, 0) @f$
 */
template <typename T0, typename T1>
XGBOOST_DEVICE std::enable_if_t<std::is_floating_point_v<T0> && std::is_floating_point_v<T1>, T0>
ThresholdL1(T0 sum_grad, T1 alpha) {
  if (sum_grad > +alpha) {
    return sum_grad - alpha;
  }
  if (sum_grad < -alpha) {
    return sum_grad + alpha;
  }
  return 0.0;
}

// calculate the cost of loss function
template <typename TrainingParams, typename T0, typename T1>
XGBOOST_DEVICE T0 CalcGainGivenWeight(TrainingParams const &p, T0 sum_grad, T0 sum_hess,
                                      T1 w) noexcept {
  return -(static_cast<T0>(2.0) * sum_grad * w + (sum_hess + p.reg_lambda) * common::Sqr(w) +
           (static_cast<T0>(2.0) * p.reg_alpha * std::abs(w)));
}

// calculate weight given the statistics
template <typename TrainingParams, typename T>
XGBOOST_DEVICE std::enable_if_t<std::is_floating_point_v<T>, T> CalcWeight(TrainingParams const &p,
                                                                           T sum_grad, T sum_hess) {
  if (sum_hess <= 0.0) {
    return 0.0;
  }
  T dw = -ThresholdL1(sum_grad, p.reg_alpha) / (sum_hess + p.reg_lambda);
  // Plain comparisons instead of fabs and copysign keep a float weight free of double
  // arithmetic, for SYCL devices without fp64 support.
  if (p.max_delta_step != 0.0f && (dw > p.max_delta_step || dw < -p.max_delta_step)) {
    dw = dw > 0 ? static_cast<T>(p.max_delta_step) : static_cast<T>(-p.max_delta_step);
  }
  return dw;
}

// calculate the cost of loss function
template <typename TrainingParams, typename T>
XGBOOST_DEVICE T CalcGain(TrainingParams const &p, T sum_grad, T sum_hess) noexcept {
  if (sum_hess <= 0.0) {
    return static_cast<T>(0.0);
  }
  if (p.max_delta_step == 0.0f) {
    if (p.reg_alpha == 0.0f) {
      return common::Sqr(sum_grad) / (sum_hess + p.reg_lambda);
    } else {
      return common::Sqr(ThresholdL1(sum_grad, p.reg_alpha)) / (sum_hess + p.reg_lambda);
    }
  } else {
    T w = CalcWeight(p, sum_grad, sum_hess);
    return CalcGainGivenWeight(p, sum_grad, sum_hess, w);
  }
}

template <typename TrainingParams, typename StatT, typename T = decltype(StatT().GetHess())>
XGBOOST_DEVICE T CalcGain(const TrainingParams &p, StatT stat) {
  return CalcGain(p, stat.GetGrad(), stat.GetHess());
}

// Used in GPU code where GradientPair is used for gradient sum, not GradStats.
template <typename TrainingParams, typename GpairT>
XGBOOST_DEVICE float CalcWeight(const TrainingParams &p, GpairT sum_grad) {
  return CalcWeight(p, sum_grad.GetGrad(), sum_grad.GetHess());
}

/**
 * @brief Weight shared by two sibling leaves when a monotone constraint forces them to be equal.
 *
 * The common value still represents two leaves, each with its own regularization penalty, so
 * the penalties of the combined statistics are doubled.
 */
template <typename TrainingParams, typename T>
XGBOOST_DEVICE T CalcPooledWeight(TrainingParams const &p, T sum_grad, T sum_hess) {
  struct PooledParams {
    float reg_lambda;
    float reg_alpha;
    float max_delta_step;
  };
  PooledParams pooled{2.0f * p.reg_lambda, 2.0f * p.reg_alpha, p.max_delta_step};
  return CalcWeight(pooled, sum_grad, sum_hess);
}
}  // namespace xgboost::tree
#endif  // XGBOOST_TREE_SPLIT_GAIN_H_
