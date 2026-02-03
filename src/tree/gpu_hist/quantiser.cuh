/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#pragma once
#include "../../common/deterministic.cuh"   // for CreateRoundingFactor
#include "../../common/device_helpers.cuh"  // for ToSpan
#include "../../common/device_vector.cuh"   // for device_vector
#include "xgboost/base.h"                   // for GradientPairPrecise, GradientPairInt64
#include "xgboost/context.h"                // for Context
#include "xgboost/data.h"                   // for MetaInfo
#include "xgboost/linalg.h"                 // for VectorView

namespace xgboost::tree {

/**
 * @brief A simple quantiser for single float values to enable deterministic summation.
 *
 * Similar to GradientQuantiser but for a single float channel.
 */
class FloatQuantiser {
 private:
  double to_fixed_point_;
  double to_floating_point_;

 public:
  FloatQuantiser(double max_abs, bst_idx_t n) {
    auto rounding = common::CreateRoundingFactor<double>(max_abs, n);
    // Use 62 bits for the mantissa (keep 1 for the sign bit)
    constexpr std::int64_t kMaxInt = static_cast<std::int64_t>(1) << 62;
    to_floating_point_ = rounding / static_cast<double>(kMaxInt);
    to_fixed_point_ = static_cast<double>(1.0) / to_floating_point_;
  }
  [[nodiscard]] double FixedPointFactor() const { return to_fixed_point_; }
  [[nodiscard]] double FloatingPointFactor() const { return to_floating_point_; }
};

/** @brief Functor for converting float to fixed-point (for use with transform iterators). */
struct ToFixedPointOp {
  double factor;
  explicit ToFixedPointOp(FloatQuantiser const& q) : factor{q.FixedPointFactor()} {}
  XGBOOST_DEVICE std::int64_t operator()(float val) const {
    return static_cast<std::int64_t>(val * factor);
  }
};

/** @brief Functor for converting fixed-point to float (for use with transform iterators). */
struct ToFloatingPointOp {
  double factor;
  explicit ToFloatingPointOp(FloatQuantiser const& q) : factor{q.FloatingPointFactor()} {}
  XGBOOST_DEVICE float operator()(std::int64_t val) const {
    return static_cast<float>(static_cast<double>(val) * factor);
  }
};
class GradientQuantiser {
 private:
  /* Convert gradient to fixed point representation. */
  GradientPairPrecise to_fixed_point_;
  /* Convert fixed point representation back to floating point. */
  GradientPairPrecise to_floating_point_;

 public:
  // Used for test
  GradientQuantiser(GradientPairPrecise to_fixed, GradientPairPrecise to_float)
      : to_fixed_point_{to_fixed}, to_floating_point_{to_float} {}
  GradientQuantiser(Context const* ctx, linalg::VectorView<GradientPair const> gpair,
                    MetaInfo const& info);
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

// For vector leaf
class MultiGradientQuantiser {
 private:
  dh::device_vector<GradientQuantiser> quantizers_;

 public:
  MultiGradientQuantiser(Context const* ctx, linalg::MatrixView<GradientPair const> gpair,
                         MetaInfo const& info);

  [[nodiscard]] auto Quantizers() const { return dh::ToSpan(this->quantizers_); }
};

void CalcQuantizedGpairs(Context const* ctx, linalg::MatrixView<GradientPair const> gpairs,
                         common::Span<GradientQuantiser const> roundings,
                         linalg::Matrix<GradientPairInt64>* p_out);
}  // namespace xgboost::tree
