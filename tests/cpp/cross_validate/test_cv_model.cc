/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>

#include <cstddef>  // for size_t
#include <memory>   // for make_unique
#include <vector>   // for vector

#include "../../../src/cross_validate/cross_validate.h"
#include "../helpers.h"  // for RandomDataGenerator
#include "xgboost/json.h"
#include "xgboost/tree_model.h"  // for RegTree

namespace xgboost::cv {
namespace {
[[nodiscard]] std::vector<gbm::TreesOneIter> MakeTrees(std::size_t k_folds,
                                                       bst_feature_t n_features) {
  std::vector<gbm::TreesOneIter> trees{k_folds};
  for (auto& fold_trees : trees) {
    fold_trees.resize(1);
    fold_trees.front().emplace_back(std::make_unique<RegTree>(1, n_features));
  }
  return trees;
}

[[nodiscard]] std::size_t NumTrees(Json const& fold) {
  auto const& booster = fold["gradient_booster"];
  auto const& model = booster["model"];
  return get<Array const>(model["trees"]).size();
}
}  // namespace

TEST(FoldModels, JsonIO) {
  std::size_t constexpr kFolds = 3;
  bst_idx_t constexpr kRows = 16;
  bst_feature_t constexpr kCols = 4;

  auto dmat = RandomDataGenerator{kRows, kCols, 0.0f}.GenerateDMatrix(true);
  FoldModels folds{kFolds, dmat};
  ASSERT_EQ(folds.KFolds(), kFolds);
  ASSERT_EQ(folds.OutputLength(0), 1);

  folds.CommitModel(MakeTrees(kFolds, kCols));

  Json model{Object{}};
  folds.SaveModel(&model);

  auto const& saved_folds = get<Array const>(model["cv_folds"]);
  ASSERT_EQ(saved_folds.size(), kFolds);
  for (auto const& fold : saved_folds) {
    auto const& obj = get<Object const>(fold);
    ASSERT_NE(obj.find("learner_model_param"), obj.cend());
    ASSERT_EQ(obj.find("properties"), obj.cend());
    ASSERT_EQ(NumTrees(fold), 1);
  }

  auto loaded = FoldModels::LoadModel(model);
  ASSERT_EQ(loaded.KFolds(), kFolds);
  ASSERT_EQ(loaded.OutputLength(0), 1);

  loaded.CommitModel(MakeTrees(kFolds, kCols));

  Json roundtrip{Object{}};
  loaded.SaveModel(&roundtrip);
  auto const& roundtrip_folds = get<Array const>(roundtrip["cv_folds"]);
  ASSERT_EQ(roundtrip_folds.size(), kFolds);
  for (auto const& fold : roundtrip_folds) {
    ASSERT_EQ(NumTrees(fold), 2);
  }
}
}  // namespace xgboost::cv
