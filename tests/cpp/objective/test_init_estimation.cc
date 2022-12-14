/**
 * Copyright 2022 by XGBoost Contributors
 */

#include <gtest/gtest.h>
#include <xgboost/linalg.h>

#include "../../src/common/linalg_op.h"
#include "../../src/objective/init_estimation.h"

namespace xgboost {
namespace obj {

void TestFitStump(Context const *ctx) {
  std::size_t constexpr kRows = 16, kTargets = 2;
  HostDeviceVector<GradientPair> gpair;
  auto &h_gpair = gpair.HostVector();
  h_gpair.resize(kRows * kTargets);
  for (std::size_t t = 0; t < kTargets; ++t) {
    for (std::size_t i = 0; i < kRows; ++i) {
      h_gpair.at(t * kRows + i) = GradientPair{static_cast<float>(i), 1};
    }
  }
  linalg::Vector<float> out;
  FitStump(ctx, gpair, kTargets, &out);
  auto h_out = out.HostView();
  for (auto it = linalg::cbegin(h_out); it != linalg::cend(h_out); ++it) {
    // sum_hess == kRows
    auto n = static_cast<float>(kRows);
    auto sum_grad = n * (n - 1) / 2;
    // ASSERT_EQ(static_cast<float>(-sum_grad / n), *it);
    std::cout << *it << std::endl;
  }
  std::cout << std::endl;
}

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
}  // namespace obj
}  // namespace xgboost
