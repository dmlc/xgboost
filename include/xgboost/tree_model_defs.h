/**
 * Copyright 2026, XGBoost contributors
 */
#pragma once

#include <xgboost/base.h>  // for bst_node_t, bst_feature_tbst_node_t
#include <xgboost/data.h>  // for FeatureType
#include <xgboost/span.h>  // for Span

#include <cstddef>  // for size_t
#include <cstdint>  // for uint32_t
#include <vector>   // for vector

namespace xgboost::tree {
using CatWordT = std::uint32_t;

/** @brief Split metadata shared by scalar and vector tree expansion. */
struct SplitInfo {
  bst_node_t nidx;
  bst_feature_t fidx;
  float cond;
  bool default_left;
  FeatureType type{FeatureType::kNumerical};
  common::Span<CatWordT const> categories{};
};

/** @brief Unscaled split weight and coverage for a node. */
template <typename Weight>
struct ExpandNodeStat {
  Weight weight;
  double sum_hess;
};

/** @brief Inputs for expanding a leaf, independent of prediction-leaf finalization. */
template <typename Weight>
struct ExpandData {
  SplitInfo split;
  ExpandNodeStat<Weight> parent;
  ExpandNodeStat<Weight> left;
  ExpandNodeStat<Weight> right;
  float loss_chg;
};

/** @brief Batch of vector expansions, with weight/category spans on the context's device. */
struct ExpandBatch {
  std::vector<ExpandData<common::Span<float const>>> nodes;
  std::size_t n_cat_words{0};

  [[nodiscard]] std::size_t Size() const { return nodes.size(); }

  void Push(ExpandData<common::Span<float const>> const& node) {
    CHECK_EQ(node.left.weight.size(), node.parent.weight.size());
    CHECK_EQ(node.right.weight.size(), node.parent.weight.size());
    CHECK(node.split.type == FeatureType::kCategorical || node.split.categories.empty());
    nodes.push_back(node);
    n_cat_words += node.split.categories.size();
  }
};
}  // namespace xgboost::tree
