/**
 * Copyright 2022-2023 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_OPTIONAL_WEIGHT_H_
#define XGBOOST_COMMON_OPTIONAL_WEIGHT_H_
#include "xgboost/base.h"                // XGBOOST_DEVICE
#include "xgboost/context.h"             // Context
#include "xgboost/host_device_vector.h"  // HostDeviceVector
#include "xgboost/span.h"                // Span

namespace xgboost::common {
struct OptionalWeights {
  Span<float const> weights;
  float dft{1.0f};  // fixme: make this compile time constant

  explicit OptionalWeights(Span<float const> w) : weights{w} {}
  explicit OptionalWeights(float w) : dft{w} {}

  XGBOOST_DEVICE float operator[](size_t i) const { return weights.empty() ? dft : weights[i]; }
  [[nodiscard]] auto Empty() const { return weights.empty(); }
  [[nodiscard]] auto Size() const { return weights.size(); }
};

inline OptionalWeights MakeOptionalWeights(Context const* ctx,
                                           HostDeviceVector<float> const& weights) {
  if (ctx->IsCUDA()) {
    weights.SetDevice(ctx->Device());
  }
  return OptionalWeights{ctx->IsCUDA() ? weights.ConstDeviceSpan() : weights.ConstHostSpan()};
}
}  // namespace xgboost::common
#endif  // XGBOOST_COMMON_OPTIONAL_WEIGHT_H_
