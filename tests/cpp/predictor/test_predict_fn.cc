/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <array>    // for array
#include <cstdint>  // for uint32_t
#include <limits>   // for numeric_limits

#include "../../../src/predictor/predict_fn.h"

namespace xgboost::predictor {
namespace {
// Minimal tree view for testing routing independently of model storage.
struct SplitView {
  bool default_left;

  bst_node_t LeftChild(bst_node_t) const { return 1; }
  bst_node_t DefaultChild(bst_node_t) const { return default_left ? 1 : 2; }
  float SplitCond(bst_node_t) const { return 0.5f; }
};

std::array<float, 4> NaNValues() {
  auto quiet = std::numeric_limits<float>::quiet_NaN();
  auto signaling = std::numeric_limits<float>::signaling_NaN();
  return {quiet, -quiet, signaling, -signaling};
}
}  // namespace

TEST(PredictFn, NumericalMissingValue) {
  RegTree::CategoricalSplitMatrix cats;
  for (bool default_left : {false, true}) {
    SplitView tree{default_left};
    for (auto value : NaNValues()) {
      EXPECT_EQ((GetNextNode<true, false>(tree, 0, value, cats)), tree.DefaultChild(0));
    }
    for (float value : {-std::numeric_limits<float>::infinity(), 0.0f, 0.5f, 1.0f,
                        std::numeric_limits<float>::infinity()}) {
      auto expected = value < 0.5f ? 1 : 2;
      EXPECT_EQ((GetNextNode<true, false>(tree, 0, value, cats)), expected);
      EXPECT_EQ((GetNextNode<false, false>(tree, 0, value, cats)), expected);
    }
  }
}

TEST(PredictFn, CategoricalMissingValue) {
  std::array<FeatureType, 1> types{FeatureType::kCategorical};
  std::array<std::uint32_t, 1> categories{};
  common::CatBitField{common::Span<std::uint32_t>{categories}}.Set(2);
  std::array<RegTree::CategoricalSplitMatrix::Segment, 1> segments{{{0, 1}}};
  RegTree::CategoricalSplitMatrix cats{types, categories, segments};
  for (bool default_left : {false, true}) {
    SplitView tree{default_left};
    for (auto value : NaNValues()) {
      EXPECT_EQ((GetNextNode<true, true>(tree, 0, value, cats)), tree.DefaultChild(0));
    }
    EXPECT_EQ((GetNextNode<true, true>(tree, 0, 2.0f, cats)), 2);
    EXPECT_EQ((GetNextNode<false, true>(tree, 0, 2.0f, cats)), 2);
    EXPECT_EQ((GetNextNode<true, true>(tree, 0, 3.0f, cats)), 1);
    EXPECT_EQ((GetNextNode<false, true>(tree, 0, 3.0f, cats)), 1);
  }
}
}  // namespace xgboost::predictor
