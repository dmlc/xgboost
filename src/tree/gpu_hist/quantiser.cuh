/**
 * Copyright 2020-2024, XGBoost Contributors
 */
#pragma once
#include "xgboost/base.h"     // for GradientPairPrecise, GradientPairInt64
#include "xgboost/context.h"  // for Context
#include "xgboost/data.h"     // for MetaInfo
#include "xgboost/span.h"     // for Span

namespace xgboost::tree {
class GradientQuantiser {
 private:
  /* Convert gradient to fixed point representation. */
  GradientPairPrecise to_fixed_point_;
  /* Convert fixed point representation back to floating point. */
  GradientPairPrecise to_floating_point_;

 public:
  GradientQuantiser(Context const* ctx, common::Span<GradientPair const> gpair,
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
}  // namespace xgboost::tree
