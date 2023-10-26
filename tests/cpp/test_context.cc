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

TEST(Context, SYCL) {
  Context ctx;
  // Default SYCL device
  {
    ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
    ASSERT_EQ(ctx.Device(), DeviceOrd::SyclDefault());
    ASSERT_EQ(ctx.Ordinal(), -1);

    std::int32_t flag{0};
    ctx.DispatchDevice([&] { flag = -1; }, [&] { flag = 1; }, [&] { flag = 2; });
    ASSERT_EQ(flag, 2);

    std::stringstream ss;
    ss << ctx.Device();
    ASSERT_EQ(ss.str(), "sycl:-1");
  }

  // SYCL device with idx
  {
    ctx.UpdateAllowUnknown(Args{{"device", "sycl:42"}});
    ASSERT_EQ(ctx.Device(), DeviceOrd::SyclDefault(42));
    ASSERT_EQ(ctx.Ordinal(), 42);

    std::int32_t flag{0};
    ctx.DispatchDevice([&] { flag = -1; }, [&] { flag = 1; }, [&] { flag = 2; });
    ASSERT_EQ(flag, 2);

    std::stringstream ss;
    ss << ctx.Device();
    ASSERT_EQ(ss.str(), "sycl:42");
  }

  // SYCL cpu
  {
    ctx.UpdateAllowUnknown(Args{{"device", "sycl:cpu"}});
    ASSERT_EQ(ctx.Device(), DeviceOrd::SyclCPU());
    ASSERT_EQ(ctx.Ordinal(), -1);

    std::int32_t flag{0};
    ctx.DispatchDevice([&] { flag = -1; }, [&] { flag = 1; }, [&] { flag = 2; });
    ASSERT_EQ(flag, 2);

    std::stringstream ss;
    ss << ctx.Device();
    ASSERT_EQ(ss.str(), "sycl:cpu:-1");
  }

  // SYCL cpu with idx
  {
    ctx.UpdateAllowUnknown(Args{{"device", "sycl:cpu:42"}});
    ASSERT_EQ(ctx.Device(), DeviceOrd::SyclCPU(42));
    ASSERT_EQ(ctx.Ordinal(), 42);

    std::int32_t flag{0};
    ctx.DispatchDevice([&] { flag = -1; }, [&] { flag = 1; }, [&] { flag = 2; });
    ASSERT_EQ(flag, 2);

    std::stringstream ss;
    ss << ctx.Device();
    ASSERT_EQ(ss.str(), "sycl:cpu:42");
  }

  // SYCL gpu
  {
    ctx.UpdateAllowUnknown(Args{{"device", "sycl:gpu"}});
    ASSERT_EQ(ctx.Device(), DeviceOrd::SyclGPU());
    ASSERT_EQ(ctx.Ordinal(), -1);

    std::int32_t flag{0};
    ctx.DispatchDevice([&] { flag = -1; }, [&] { flag = 1; }, [&] { flag = 2; });
    ASSERT_EQ(flag, 2);

    std::stringstream ss;
    ss << ctx.Device();
    ASSERT_EQ(ss.str(), "sycl:gpu:-1");
  }

  // SYCL gpu with idx
  {
    ctx.UpdateAllowUnknown(Args{{"device", "sycl:gpu:42"}});
    ASSERT_EQ(ctx.Device(), DeviceOrd::SyclGPU(42));
    ASSERT_EQ(ctx.Ordinal(), 42);

    std::int32_t flag{0};
    ctx.DispatchDevice([&] { flag = -1; }, [&] { flag = 1; }, [&] { flag = 2; });
    ASSERT_EQ(flag, 2);

    std::stringstream ss;
    ss << ctx.Device();
    ASSERT_EQ(ss.str(), "sycl:gpu:42");
  }
}
}  // namespace xgboost
