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
[[nodiscard]] std::vector<gbm::TreesOneIter> MakeTrees(std::size_t n_units,
                                                       bst_feature_t n_features) {
  std::vector<gbm::TreesOneIter> trees{n_units};
  for (auto& unit_trees : trees) {
    unit_trees.resize(1);
    auto tree = std::make_unique<RegTree>(1, n_features, true);
    std::vector<float> weight{0.0f};
    tree->SetRoot(linalg::MakeVec(weight), 0.0f);
    tree->GetMultiTargetTree()->SetLeaves();
    unit_trees.front().emplace_back(std::move(tree));
  }
  return trees;
}

[[nodiscard]] std::size_t NumTrees(Json const& unit) {
  auto const& booster = unit["gradient_booster"];
  auto const& model = booster["model"];
  return get<Array const>(model["trees"]).size();
}

void CheckUnitShape(Json const& unit, std::size_t n_trees) {
  auto const& obj = get<Object const>(unit);
  ASSERT_NE(obj.find("learner_model_param"), obj.cend());
  ASSERT_NE(obj.find("objective"), obj.cend());
  ASSERT_EQ(obj.find("properties"), obj.cend());
  ASSERT_EQ(NumTrees(unit), n_trees);
}

// The folds are an array. The refit model is not a fold, so it is a sibling key, absent when
// the run has none, which is how `LoadModel` recovers the layout.
void CheckModelShape(Json const& model, std::size_t k_folds, bool refit, std::size_t n_trees) {
  auto const& saved_folds = get<Array const>(model["cv_folds"]);
  ASSERT_EQ(saved_folds.size(), k_folds);
  for (auto const& fold : saved_folds) {
    CheckUnitShape(fold, n_trees);
  }

  auto const& obj = get<Object const>(model);
  ASSERT_EQ(obj.find("refit") != obj.cend(), refit);
  if (refit) {
    CheckUnitShape(model["refit"], n_trees);
  }
}

class FoldModelsIO : public ::testing::TestWithParam<bool> {};
}  // namespace

TEST_P(FoldModelsIO, Json) {
  bool const refit = this->GetParam();
  std::size_t constexpr kFolds = 3;
  bst_idx_t constexpr kRows = 16;
  bst_feature_t constexpr kCols = 4;
  auto const n_units = kFolds + static_cast<std::size_t>(refit);

  auto dmat = RandomDataGenerator{kRows, kCols, 0.0f}.GenerateDMatrix(true);
  FoldModels folds{kFolds, dmat, refit};
  ASSERT_EQ(folds.KFolds(), kFolds);
  ASSERT_EQ(folds.NumUnits(), n_units);
  ASSERT_EQ(folds.HasRefit(), refit);
  if (refit) {
    ASSERT_EQ(folds.RefitIdx(), kFolds);  // The refit model is the last unit.
  }
  ASSERT_EQ(folds.OutputLength(n_units - 1), 1);
  ASSERT_EQ(folds.BoostedRounds(), 0);

  // Every unit advances in lockstep, so one round is one tree per unit.
  folds.CommitModel(MakeTrees(n_units, kCols));
  ASSERT_EQ(folds.BoostedRounds(), 1);

  Json model{Object{}};
  folds.SaveModel(&model);
  CheckModelShape(model, kFolds, refit, 1);

  auto loaded = FoldModels::LoadModel(model);
  ASSERT_EQ(loaded.KFolds(), kFolds);
  ASSERT_EQ(loaded.NumUnits(), n_units);
  ASSERT_EQ(loaded.HasRefit(), refit);
  ASSERT_EQ(loaded.OutputLength(n_units - 1), 1);
  ASSERT_EQ(loaded.BoostedRounds(), 1);

  loaded.CommitModel(MakeTrees(n_units, kCols));
  ASSERT_EQ(loaded.BoostedRounds(), 2);

  Json roundtrip{Object{}};
  loaded.SaveModel(&roundtrip);
  CheckModelShape(roundtrip, kFolds, refit, 2);
}

INSTANTIATE_TEST_SUITE_P(FoldModels, FoldModelsIO, ::testing::Bool());
}  // namespace xgboost::cv
