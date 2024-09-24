/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>  // for Args
#include <xgboost/context.h>
#include <xgboost/json.h>  // for FromJson, ToJson

#include <string>  // for string, to_string

#include "../../src/common/cuda_rt_utils.h"  // for AllVisibleGPUs

namespace xgboost {
namespace {
void TestCUDA(Context const& ctx, bst_d_ordinal_t ord) {
  ASSERT_EQ(ctx.Device().ordinal, ord);
  ASSERT_EQ(ctx.DeviceName(), "cuda:" + std::to_string(ord));
  ASSERT_EQ(ctx.Ordinal(), ord);
  ASSERT_TRUE(ctx.IsCUDA());
  ASSERT_FALSE(ctx.IsCPU());
  ASSERT_EQ(ctx.Device(), DeviceOrd::CUDA(ord));

  Json jctx{ToJson(ctx)};
  Context new_ctx;
  FromJson(jctx, &new_ctx);
  ASSERT_EQ(new_ctx.Device(), ctx.Device());
  ASSERT_EQ(new_ctx.Ordinal(), ctx.Ordinal());
}
}  // namespace

TEST(Context, MGPUDeviceOrdinal) {
  Context ctx;
  auto n_vis = curt::AllVisibleGPUs();
  auto ord = n_vis - 1;

  std::string device = "cuda:" + std::to_string(ord);
  ctx.UpdateAllowUnknown(Args{{"device", device}});
  TestCUDA(ctx, ord);

  device = "cuda:" + std::to_string(1001);
  ctx.UpdateAllowUnknown(Args{{"device", device}});
  ord = 1001 % n_vis;

  TestCUDA(ctx, ord);

  std::int32_t flag{0};
  ctx.DispatchDevice([&] { flag = -1; }, [&] { flag = 1; });
  ASSERT_EQ(flag, 1);

  Context new_ctx = ctx;
  TestCUDA(new_ctx, ctx.Ordinal());

  auto cpu_ctx = ctx.MakeCPU();
  ASSERT_TRUE(cpu_ctx.IsCPU());
  ASSERT_EQ(cpu_ctx.Ordinal(), DeviceOrd::CPUOrdinal());
  ASSERT_EQ(cpu_ctx.Device(), DeviceOrd::CPU());

  auto cuda_ctx = cpu_ctx.MakeCUDA(ctx.Ordinal());
  TestCUDA(cuda_ctx, ctx.Ordinal());

  cuda_ctx.UpdateAllowUnknown(Args{{"fail_on_invalid_gpu_id", "true"}});
  ASSERT_THROW({ cuda_ctx.UpdateAllowUnknown(Args{{"device", "cuda:9999"}}); }, dmlc::Error);
  cuda_ctx.UpdateAllowUnknown(Args{{"device", "cuda:00"}});
  ASSERT_EQ(cuda_ctx.Ordinal(), 0);

  ctx.UpdateAllowUnknown(Args{{"device", "cpu"}});
  // Test alias
  ctx.UpdateAllowUnknown(Args{{"device", "gpu:0"}});
  TestCUDA(ctx, 0);
  ctx.UpdateAllowUnknown(Args{{"device", "gpu"}});
  TestCUDA(ctx, 0);

  // Test the thread local memory in dmlc is not linking different instances together.
  cpu_ctx.UpdateAllowUnknown(Args{{"device", "cpu"}});
  TestCUDA(ctx, 0);
  ctx.UpdateAllowUnknown(Args{});
  TestCUDA(ctx, 0);
}

TEST(Context, MGPUId) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"gpu_id", "0"}});
  TestCUDA(ctx, 0);

  auto n_vis = curt::AllVisibleGPUs();
  auto ord = n_vis - 1;
  ctx.UpdateAllowUnknown(Args{{"gpu_id", std::to_string(ord)}});
  TestCUDA(ctx, ord);

  auto device = "cuda:" + std::to_string(1001);
  ctx.UpdateAllowUnknown(Args{{"device", device}});
  ord = 1001 % n_vis;
  TestCUDA(ctx, ord);

  ctx.UpdateAllowUnknown(Args{{"gpu_id", "-1"}});
  ASSERT_EQ(ctx.Device(), DeviceOrd::CPU());
}
}  // namespace xgboost
