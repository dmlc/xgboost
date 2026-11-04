/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <array>    // for array
#include <cmath>    // for hypot
#include <cstddef>  // for size_t
#include <vector>   // for vector

#include "../../../src/common/kernel.h"  // for DispatchKernel
#include "../../../src/gbm/random_projection.h"
#include "../collective/test_worker.h"  // for TestDistributedGlobal
#include "../helpers.h"                 // for MakeCUDACtx
#include "xgboost/base.h"
#include "xgboost/context.h"
#include "xgboost/linalg.h"

namespace xgboost::gbm {
TEST(RandomProjection, MakeRandomProjection) {
  Context ctx;
  auto basis = MakeRandomProjection(&ctx, 2, 2);
  auto q = basis.HostView();
  // Random values can vary, but the columns must be unit vectors at right angles.
  EXPECT_NEAR(std::hypot(q(0, 0), q(1, 0)), 1.0, 1e-12);
  EXPECT_NEAR(std::hypot(q(0, 1), q(1, 1)), 1.0, 1e-12);
  EXPECT_NEAR(q(0, 0) * q(0, 1) + q(1, 0) * q(1, 1), 0.0, 1e-12);
}

TEST(RandomProjection, DistributedBasis) {
  std::array<std::vector<double>, 2> bases;
  collective::TestDistributedGlobal(2, [&] {
    auto rank = collective::GetRank();
    Context ctx;
    // Sampling different shards can leave workers with different RNG states.
    ctx.Rng().seed(rank);
    auto basis = MakeRandomProjection(&ctx, 16, 4);
    bases[rank] = basis.Data()->ConstHostVector();
  });
  EXPECT_EQ(bases[0], bases[1]);
}

namespace {
void TestProjection(Context const* ctx) {
  std::array const gradients{GradientPair{2.0f, 1.0f}, GradientPair{4.0f, 2.0f},
                             GradientPair{6.0f, 3.0f}, GradientPair{8.0f, 4.0f}};
  // Q's columns are (1, 1, 1, 1) / 2 and (1, 1, -1, -1) / 2: orthogonal unit vectors.
  // Projected gradients: (2 + 4 + 6 + 8) / 2 = 10 and (2 + 4 - 6 - 8) / 2 = -4.
  // Diagonal Hessians use squared coefficients, so both are (1 + 2 + 3 + 4) / 4 = 2.5.
  std::array const expected{GradientPair{10.0f, 2.5f}, GradientPair{-4.0f, 2.5f}};
  auto n_targets = gradients.size();
  auto n_components = expected.size();
  linalg::Matrix<double> basis{
      {0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5}, {n_targets, n_components}, ctx->Device()};
  linalg::Matrix<GradientPair> input{
      gradients.cbegin(), gradients.cend(), {std::size_t{1}, n_targets}, ctx->Device()};
  auto output = linalg::Empty<GradientPair>(ctx, 1, n_components);
  common::DispatchKernel<ProjectGradientKernel>(
      ctx, input.View(ctx->Device()), basis.View(ctx->Device()), output.View(ctx->Device()));
  if (ctx->IsCUDA()) {
    EXPECT_FALSE(input.Data()->HostCanRead());
    EXPECT_FALSE(output.Data()->HostCanRead());
  }
  auto out = output.HostView();
  for (std::size_t t = 0; t < n_components; ++t) {
    EXPECT_FLOAT_EQ(out(0, t).GetGrad(), expected[t].GetGrad());
    EXPECT_FLOAT_EQ(out(0, t).GetHess(), expected[t].GetHess());
  }
}
}  // namespace

TEST(RandomProjection, CPU) {
  Context ctx;
  TestProjection(&ctx);
}

#if defined(XGBOOST_USE_CUDA)
TEST(RandomProjection, CUDA) {
  auto ctx = MakeCUDACtx(0);
  TestProjection(&ctx);
}
#endif  // defined(XGBOOST_USE_CUDA)
}  // namespace xgboost::gbm
