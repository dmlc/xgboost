/**
 * Copyright 2022-2024, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/linalg.h>

#include "../../src/common/linalg_op.h"
#include "../../src/tree/fit_stump.h"
#include "../collective/test_worker.h"  // for TestDistributedGlobal
#include "../helpers.h"

namespace xgboost::tree {
namespace {
void TestFitStump(Context const *ctx, DataSplitMode split = DataSplitMode::kRow) {
  std::size_t constexpr kRows = 16, kTargets = 2;
  linalg::Matrix<GradientPair> gpair;
  gpair.SetDevice(ctx->Device());
  gpair.Reshape(kRows, kTargets);
  auto h_gpair = gpair.HostView();
  for (std::size_t i = 0; i < kRows; ++i) {
    for (std::size_t t = 0; t < kTargets; ++t) {
      h_gpair(i, t) = GradientPair{static_cast<float>(i), 1};
    }
  }
  linalg::Vector<float> out;
  MetaInfo info;
  info.data_split_mode = split;
  FitStump(ctx, info, gpair, kTargets, &out);
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
  ctx.UpdateAllowUnknown(Args{{"device", "cuda"}});
  TestFitStump(&ctx);
}
#endif  // defined(XGBOOST_USE_CUDA)

TEST(InitEstimation, FitStumpColumnSplit) {
  Context ctx;
  auto constexpr kWorldSize{3};
  collective::TestDistributedGlobal(kWorldSize, [&] { TestFitStump(&ctx, DataSplitMode::kCol); });
}
}  // namespace xgboost::tree
