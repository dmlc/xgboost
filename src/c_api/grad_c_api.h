/**
 * Copyright 2025, XGBoost Contributors
 */
#pragma once

#include "xgboost/context.h"
#include "xgboost/gradient.h"

typedef void *GradientContainerHandle;  // NOLINT(*)

namespace xgboost {
struct GradientContainerWithCtx {
  Context const *ctx;
  GradientContainer gpairs;

  explicit GradientContainerWithCtx(Context const *ctx, common::Span<std::size_t const, 2> shape)
      : ctx{ctx}, gpairs{ctx, shape} {}
};

inline auto CastGradientContainerHandle(GradientContainerHandle handle) {
  CHECK(handle) << "Invalid gradient container handle.";
  return static_cast<GradientContainerWithCtx *>(handle);
}
}  // namespace xgboost
