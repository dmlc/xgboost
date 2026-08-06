/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <cstdint>  // for int32_t

#include "../../../src/common/kernel.h"
#include "xgboost/context.h"  // for Context

namespace xgboost::common {
namespace {
struct FallbackKernel {
  using Signature = std::int32_t(Context const*, std::int32_t);
};

std::int32_t FallbackCPU(Context const* ctx, std::int32_t value) {
  return ctx->IsCPU() ? value : -1;
}

KernelRegistration<FallbackKernel> const register_fallback_cpu{DeviceOrd::kCPU, &FallbackCPU};
}  // namespace

TEST(Kernel, CPUFallback) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", DeviceSym::SyclDefault()}});

  EXPECT_EQ(DispatchKernel<FallbackKernel>(&ctx, 42), 42);
}
}  // namespace xgboost::common
