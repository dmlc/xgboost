/**
 * Copyright 2022 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/linalg.h>

#include "../../src/common/linalg_op.h"
#include "../../src/tree/fit_stump.h"

namespace xgboost {
namespace tree {
namespace {
void TestFitStump(Context const *ctx) {
  std::size_t constexpr kRows = 16, kTargets = 2;
  HostDeviceVector<GradientPair> gpair;
  auto &h_gpair = gpair.HostVector();
  h_gpair.resize(kRows * kTargets);
  for (std::size_t i = 0; i < kRows; ++i) {
    for (std::size_t t = 0; t < kTargets; ++t) {
      h_gpair.at(i * kTargets + t) = GradientPair{static_cast<float>(i), 1};
    }
  }
  linalg::Vector<float> out;
  FitStump(ctx, gpair, kTargets, &out);
  auto h_out = out.HostView();
  for (auto it = linalg::cbegin(h_out); it != linalg::cend(h_out); ++it) {
    // sum_hess == kRows
    auto n = static_cast<float>(kRows);
    auto sum_grad = n * (n - 1) / 2;
    ASSERT_EQ(static_cast<float>(-sum_grad / n), *it);
  }
}
}  // anonymous namespace

TEST(InitEstimation, FitStump) {
  Context ctx;
  TestFitStump(&ctx);
}

#if defined(XGBOOST_USE_CUDA)
TEST(InitEstimation, GPUFitStump) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"gpu_id", "0"}});
  TestFitStump(&ctx);
}
#endif  // defined(XGBOOST_USE_CUDA)
}  // namespace tree
}  // namespace xgboost
