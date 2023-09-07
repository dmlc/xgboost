/**
 * Copyright 2023, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>
#include <xgboost/context.h>

namespace xgboost {
TEST(Context, CPU) {
  Context ctx;
  ASSERT_EQ(ctx.Device(), DeviceOrd::CPU());
  ASSERT_EQ(ctx.Ordinal(), Context::kCpuId);

  std::int32_t flag{0};
  ctx.DispatchDevice([&] { flag = -1; }, [&] { flag = 1; });
  ASSERT_EQ(flag, -1);

  ASSERT_THROW(ctx.UpdateAllowUnknown(Args{{"device", "oops"}}), dmlc::Error);
  ASSERT_THROW(ctx.UpdateAllowUnknown(Args{{"device", "-1"}}), dmlc::Error);
  ASSERT_THROW(ctx.UpdateAllowUnknown(Args{{"device", "CPU"}}), dmlc::Error);
  ASSERT_THROW(ctx.UpdateAllowUnknown(Args{{"device", "CUDA"}}), dmlc::Error);
  ASSERT_THROW(ctx.UpdateAllowUnknown(Args{{"device", "CPU:0"}}), dmlc::Error);
  ASSERT_THROW(ctx.UpdateAllowUnknown(Args{{"device", "gpu:+0"}}), dmlc::Error);
  ASSERT_THROW(ctx.UpdateAllowUnknown(Args{{"device", "gpu:0-"}}), dmlc::Error);
  ASSERT_THROW(ctx.UpdateAllowUnknown(Args{{"device", "gpu:"}}), dmlc::Error);
  ASSERT_THROW(ctx.UpdateAllowUnknown(Args{{"device", ":"}}), dmlc::Error);
  ASSERT_THROW(ctx.UpdateAllowUnknown(Args{{"device", ":gpu"}}), dmlc::Error);
  ASSERT_THROW(ctx.UpdateAllowUnknown(Args{{"device", ":0"}}), dmlc::Error);
  ASSERT_THROW(ctx.UpdateAllowUnknown(Args{{"device", ""}}), dmlc::Error);
}
}  // namespace xgboost
