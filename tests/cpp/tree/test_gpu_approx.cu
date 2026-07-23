/**
 * Copyright 2024-2026, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/gradient.h>      // for GradientContainer
#include <xgboost/task.h>          // for ObjInfo
#include <xgboost/tree_model.h>    // for RegTree
#include <xgboost/tree_updater.h>  // for TreeUpdater

#include "../../../src/tree/param.h"  // for TrainParam
#include "../helpers.h"               // for GenerateRandomGradients

namespace xgboost::tree {
namespace {
RegTree GetApproxTree(Context const* ctx, DMatrix* dmat) {
  ObjInfo task{ObjInfo::kRegression};
  std::unique_ptr<TreeUpdater> approx_maker{TreeUpdater::Create("grow_gpu_approx", ctx, &task)};
  approx_maker->Configure(Args{});

  TrainParam param;
  param.UpdateAllowUnknown(Args{});

  auto gpair = GenerateRandomGradients(ctx, dmat->Info().num_row_, 1);

  std::vector<HostDeviceVector<bst_node_t>> position(1);
  RegTree tree;
  approx_maker->Update(&param, &gpair, dmat, common::Span<HostDeviceVector<bst_node_t>>{position},
                       {&tree});
  return tree;
}
}  // anonymous namespace

TEST(GpuApprox, Basic) {
  constexpr bst_idx_t kRows = 32;
  constexpr bst_feature_t kCols = 16;

  auto ctx = MakeCUDACtx(0);
  auto dmat = RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix(true);
  auto tree = GetApproxTree(&ctx, dmat.get());

  ASSERT_EQ(tree.NumNodes(), 7);
}
}  // namespace xgboost::tree
