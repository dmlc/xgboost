/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#pragma once

#include <xgboost/data.h>          // for FeatureType, DMatrix
#include <xgboost/tree_model.h>    // for RegTree
#include <xgboost/tree_updater.h>  // for TreeUpdater

#include <cstddef>  // for size_t
#include <memory>   // for shared_ptr
#include <vector>   // for vector

#include "../../../src/tree/param.h"    // for TrainParam
#include "../collective/test_worker.h"  // for TestDistributedGlobal
#include "../helpers.h"                 // for RandomDataGenerator

namespace xgboost::tree {
inline std::shared_ptr<DMatrix> GenerateCatDMatrix(std::size_t rows, std::size_t cols,
                                                   float sparsity, bool categorical) {
  if (categorical) {
    std::vector<FeatureType> ft(cols);
    for (size_t i = 0; i < ft.size(); ++i) {
      ft[i] = (i % 3 == 0) ? FeatureType::kNumerical : FeatureType::kCategorical;
    }
    return RandomDataGenerator(rows, cols, 0.6f).Seed(3).Type(ft).MaxCategory(17).GenerateDMatrix();
  } else {
    return RandomDataGenerator{rows, cols, 0.6f}.Seed(3).GenerateDMatrix();
  }
}

inline void TestColumnSplit(bst_target_t n_targets, bool categorical, std::string name,
                            float sparsity) {
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

  auto verify = [&] {
    Context ctx;
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

  auto constexpr kWorldSize = 2;
  collective::TestDistributedGlobal(kWorldSize, [&] { verify(); });
}
}  // namespace xgboost::tree
