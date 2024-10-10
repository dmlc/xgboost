/**
 * Copyright 2024, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/json.h>          // for Json
#include <xgboost/task.h>          // for ObjInfo
#include <xgboost/tree_model.h>    // for RegTree
#include <xgboost/tree_updater.h>  // for TreeUpdater

#include "../../../src/tree/param.h"    // for TrainParam
#include "../collective/test_worker.h"  // for BaseMGPUTest
#include "../helpers.h"                 // for GenerateRandomGradients

namespace xgboost::tree {
namespace {
RegTree GetApproxTree(Context const* ctx, DMatrix* dmat) {
  ObjInfo task{ObjInfo::kRegression};
  std::unique_ptr<TreeUpdater> approx_maker{TreeUpdater::Create("grow_gpu_approx", ctx, &task)};
  approx_maker->Configure(Args{});

  TrainParam param;
  param.UpdateAllowUnknown(Args{});

  linalg::Matrix<GradientPair> gpair({dmat->Info().num_row_}, ctx->Device());
  gpair.Data()->Copy(GenerateRandomGradients(dmat->Info().num_row_));

  std::vector<HostDeviceVector<bst_node_t>> position(1);
  RegTree tree;
  approx_maker->Update(&param, &gpair, dmat, common::Span<HostDeviceVector<bst_node_t>>{position},
                       {&tree});
  return tree;
}

void VerifyApproxColumnSplit(bst_idx_t rows, bst_feature_t cols, RegTree const& expected_tree) {
  auto ctx = MakeCUDACtx(DistGpuIdx());

  auto Xy = RandomDataGenerator{rows, cols, 0}.GenerateDMatrix(true);
  auto const world_size = collective::GetWorldSize();
  auto const rank = collective::GetRank();
  std::unique_ptr<DMatrix> sliced{Xy->SliceCol(world_size, rank)};

  RegTree tree = GetApproxTree(&ctx, sliced.get());

  Json json{Object{}};
  tree.SaveModel(&json);
  Json expected_json{Object{}};
  expected_tree.SaveModel(&expected_json);
  ASSERT_EQ(json, expected_json);
}
}  // anonymous namespace

class MGPUApproxTest : public collective::BaseMGPUTest {};

TEST_F(MGPUApproxTest, GPUApproxColumnSplit) {
  auto constexpr kRows = 32;
  auto constexpr kCols = 16;

  Context ctx(MakeCUDACtx(0));
  auto dmat = RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix(true);
  RegTree expected_tree = GetApproxTree(&ctx, dmat.get());

  this->DoTest([&] { VerifyApproxColumnSplit(kRows, kCols, expected_tree); }, true);
  this->DoTest([&] { VerifyApproxColumnSplit(kRows, kCols, expected_tree); }, false);
}
}  // namespace xgboost::tree
