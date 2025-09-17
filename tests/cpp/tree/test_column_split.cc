/**
 * Copyright 2024, XGBoost Contributors
 */
#include "test_column_split.h"

#include <gtest/gtest.h>
#include <xgboost/tree_model.h>    // for RegTree
#include <xgboost/tree_updater.h>  // for TreeUpdater

#include <vector>  // for vector

#include "../../../src/tree/param.h"    // for TrainParam
#include "../collective/test_worker.h"  // for TestDistributedGlobal

namespace xgboost::tree {
void TestColumnSplit(bst_target_t n_targets, bool categorical, std::string name, float sparsity) {
  auto constexpr kRows = 32;
  auto constexpr kCols = 16;

  RegTree expected_tree{n_targets, static_cast<bst_feature_t>(kCols)};
  ObjInfo task{ObjInfo::kRegression};
  Context ctx;
  {
    auto p_dmat = GenerateCatDMatrix(kRows, kCols, sparsity, categorical);
    auto gpair = GenerateRandomGradients(&ctx, kRows, n_targets);
    std::unique_ptr<TreeUpdater> updater{TreeUpdater::Create(name, &ctx, &task)};
    std::vector<HostDeviceVector<bst_node_t>> position(1);
    TrainParam param;
    param.Init(Args{});
    updater->Configure(Args{});
    updater->Update(&param, &gpair, p_dmat.get(), position, {&expected_tree});
  }

  auto constexpr kWorldSize = 2;

  auto verify = [&] {
    Context ctx;
    collective::GetWorkerLocalThreads(kWorldSize, &ctx);

    auto p_dmat = GenerateCatDMatrix(kRows, kCols, sparsity, categorical);
    auto gpair = GenerateRandomGradients(&ctx, kRows, n_targets);

    ObjInfo task{ObjInfo::kRegression};
    std::unique_ptr<TreeUpdater> updater{TreeUpdater::Create(name, &ctx, &task)};
    std::vector<HostDeviceVector<bst_node_t>> position(1);

    std::unique_ptr<DMatrix> sliced{
        p_dmat->SliceCol(collective::GetWorldSize(), collective::GetRank())};

    RegTree tree{n_targets, static_cast<bst_feature_t>(kCols)};
    TrainParam param;
    param.Init(Args{});
    updater->Configure(Args{});
    updater->Update(&param, &gpair, sliced.get(), position, {&tree});

    Json json{Object{}};
    tree.SaveModel(&json);
    Json expected_json{Object{}};
    expected_tree.SaveModel(&expected_json);
    ASSERT_EQ(json, expected_json);
  };

  collective::TestDistributedGlobal(kWorldSize, [&] { verify(); });
}
}  // namespace xgboost::tree
