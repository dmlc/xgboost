/**
 * Copyright 2023, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>
#include <xgboost/context.h>

#include <sstream>

namespace xgboost {
TEST(Context, CPU) {
  Context ctx;
  ASSERT_EQ(ctx.Device(), DeviceOrd::CPU());
  ASSERT_EQ(ctx.Ordinal(), DeviceOrd::CPUOrdinal());

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

  std::stringstream ss;
  ss << ctx.Device();
  ASSERT_EQ(ss.str(), "cpu");
}

TEST(Context, ErrorInit) {
  Context ctx;
  ASSERT_THROW({ ctx.Init({{"foo", "bar"}}); }, dmlc::Error);
  try {
    ctx.Init({{"foo", "bar"}});
  } catch (dmlc::Error const& e) {
    auto msg = std::string{e.what()};
    ASSERT_NE(msg.find("foo"), std::string::npos);
  }
}
}  // namespace xgboost
