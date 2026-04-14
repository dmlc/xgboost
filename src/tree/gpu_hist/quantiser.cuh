/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#pragma once
#include <vector>  // for vector

#include "../../common/deterministic.cuh"   // for CreateRoundingFactor
#include "../../common/device_helpers.cuh"  // for ToSpan
#include "../../common/device_vector.cuh"   // for device_vector, DeviceUVector
#include "xgboost/base.h"                   // for GradientPairPrecise, GradientPairInt64
#include "xgboost/context.h"                // for Context
#include "xgboost/data.h"                   // for MetaInfo
#include "xgboost/linalg.h"                 // for VectorView, MatrixView

namespace xgboost::tree {

/**
 * @brief A simple quantiser for single float values to enable deterministic summation.
 *
 * Similar to GradientQuantiser but for a single float channel.
 */
struct FloatQuantiser {
  double to_fixed_point;
  double to_floating_point;
  FloatQuantiser(double max_abs, bst_idx_t n) {
    auto rounding = common::CreateRoundingFactor<double>(max_abs, n);
    // See the gradient quantizer for details.
    constexpr std::int64_t kMaxInt = static_cast<std::int64_t>(1) << 62;
    to_floating_point = rounding / static_cast<double>(kMaxInt);
    to_fixed_point = static_cast<double>(1.0) / to_floating_point;
  }
};
// Functors that can be easily passed into thrust algorithms
struct ToFixedPointOp {
  double factor;
  explicit ToFixedPointOp(FloatQuantiser const& q) : factor{q.to_fixed_point} {}
  XGBOOST_DEVICE std::int64_t operator()(double val) const {
    return static_cast<std::int64_t>(val * factor);
  }
};
struct ToFloatingPointOp {
  double factor;
  explicit ToFloatingPointOp(FloatQuantiser const& q) : factor{q.to_floating_point} {}
  XGBOOST_DEVICE double operator()(std::int64_t val) const {
    return static_cast<double>(val) * factor;
  }
};

/**
 * @brief Per-target quantiser for converting gradients between floating-point and
 *        fixed-point representations.
 */
class GradientQuantiser {
 private:
  /* Convert gradient to fixed point representation. */
  GradientPairPrecise to_fixed_point_;
  /* Convert fixed point representation back to floating point. */
  GradientPairPrecise to_floating_point_;

 public:
  GradientQuantiser() = default;
  // Used for test
  GradientQuantiser(GradientPairPrecise to_fixed, GradientPairPrecise to_float)
      : to_fixed_point_{to_fixed}, to_floating_point_{to_float} {}
  [[nodiscard]] XGBOOST_DEVICE GradientPairInt64 ToFixedPoint(GradientPair const& gpair) const {
    auto adjusted = GradientPairInt64(gpair.GetGrad() * to_fixed_point_.GetGrad(),
                                      gpair.GetHess() * to_fixed_point_.GetHess());
    return adjusted;
  }
  [[nodiscard]] XGBOOST_DEVICE GradientPairInt64
  ToFixedPoint(GradientPairPrecise const& gpair) const {
    auto adjusted = GradientPairInt64(gpair.GetGrad() * to_fixed_point_.GetGrad(),
                                      gpair.GetHess() * to_fixed_point_.GetHess());
    return adjusted;
  }
  [[nodiscard]] XGBOOST_DEVICE GradientPairPrecise
  ToFloatingPoint(const GradientPairInt64& gpair) const {
    auto g = gpair.GetQuantisedGrad() * to_floating_point_.GetGrad();
    auto h = gpair.GetQuantisedHess() * to_floating_point_.GetHess();
    return {g, h};
  }
};

/**
 * @brief Unified quantiser group for single-target and multi-target gradient quantisation.
 */
class GradientQuantiserGroup {
 private:
  std::vector<GradientQuantiser> h_quantizers_;
  dh::DeviceUVector<GradientQuantiser> d_quantizers_;

 public:
  /** @brief Construct from a gradient matrix (n_samples x n_targets). */
  GradientQuantiserGroup(Context const* ctx, linalg::MatrixView<GradientPair const> gpair,
                         MetaInfo const& info);
  /** @brief Convenience constructor from a vector (single-target). */
  GradientQuantiserGroup(Context const* ctx, linalg::VectorView<GradientPair const> gpair,
                         MetaInfo const& info);

  [[nodiscard]] common::Span<GradientQuantiser const> DeviceSpan() const {
    return dh::ToSpan(this->d_quantizers_);
  }
  [[nodiscard]] GradientQuantiser const& operator[](bst_target_t t) const {
    return this->h_quantizers_[t];
  }
  [[nodiscard]] bst_target_t Size() const { return this->h_quantizers_.size(); }
};

void CalcQuantizedGpairs(Context const* ctx, linalg::MatrixView<GradientPair const> gpairs,
                         common::Span<GradientQuantiser const> roundings,
                         linalg::Matrix<GradientPairInt64>* p_out);
}  // namespace xgboost::tree
