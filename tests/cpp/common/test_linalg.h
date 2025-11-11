/**
 * Copyright 2025, XGBoost Contributors
 */
#pragma once

#include <gtest/gtest.h>
#include <xgboost/context.h>
#include <xgboost/linalg.h>  // for Vector

#include <numeric>  // for iota
#include <vector>   // for vector

#include "../../../src/common/linalg_op.h"

namespace xgboost::linalg {
template <typename Fn>
void TestLinalgDispatch(Context const* ctx, Fn&& fn) {
  std::vector<double> data(128, 0);
  std::iota(data.begin(), data.end(), 0.0);
  Vector<double> vec(data.begin(), data.end(), {data.size()}, DeviceOrd::CPU());

  TransformKernel(ctx, vec.View(ctx->Device()), [=] XGBOOST_DEVICE(double v) { return fn(v); });
  auto h_v = vec.HostView();
  for (std::size_t i = 0; i < h_v.Size(); ++i) {
    ASSERT_EQ(h_v(i), fn(i));
  }
}
}  // namespace xgboost::linalg
