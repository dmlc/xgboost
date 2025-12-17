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

  explicit GradientContainerWithCtx(Context const *ctx) : ctx{ctx} {}
};

inline auto CastGradientContainerHandle(GradientContainerHandle handle) {
  CHECK(handle) << "Invalid gradient container handle.";
  return static_cast<GradientContainerWithCtx *>(handle);
}
}  // namespace xgboost
