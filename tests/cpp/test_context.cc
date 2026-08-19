/**
 * Copyright 2023, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>
#include <xgboost/context.h>

#include <random>  // for mt19937
#include <sstream>
#include <string>

#include "../../src/common/random.h"  // for SaveRng
#include "xgboost/json.h"             // for Json, Object, String, get

namespace xgboost {
// The RNG state must survive a config round trip, and must be written in a form that does
// not depend on the standard library that wrote it. The text produced by `operator<<` for
// `std::mt19937` is not: libstdc++, libc++ and the MSVC STL each spell it differently, so
// a snapshot written on Linux could not be read on Windows.
// https://github.com/dmlc/xgboost/issues/12459
TEST(Context, RngStateRoundTrip) {
  Context ctx;
  ctx.Init({{"seed", "1994"}});
  for (std::size_t i = 0; i < 13; ++i) {
    static_cast<void>(ctx.Rng()());
  }

  Context loaded;
  loaded.FromJson(ctx.ToJson());
  ASSERT_EQ(loaded.seed, 1994);
  ASSERT_EQ(loaded.Rng().NumAdvanced(), ctx.Rng().NumAdvanced());
  for (std::size_t i = 0; i < 16; ++i) {
    ASSERT_EQ(loaded.Rng()(), ctx.Rng()());
  }
}

TEST(Context, LegacyRngState) {
  Context ctx;
  ctx.Init({{"seed", "1994"}});
  auto j = ctx.ToJson();

  // Overwrite with the state as older versions wrote it.
  std::stringstream ss;
  ss << std::hex << std::mt19937{7};
  j["rng_state"] = String{ss.str()};

  // It must be refused rather than misread, and the engine falls back to the seed.
  Context loaded;
  loaded.FromJson(j);
  ASSERT_EQ(loaded.seed, 1994);
  ASSERT_EQ(loaded.Rng().Seed(), 1994u);
  ASSERT_EQ(loaded.Rng().NumAdvanced(), 0ul);
}

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
