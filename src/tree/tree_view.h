/**
 * Copyright 2025, XGBoost Contributors
 */
#pragma once
#include <stack>    // for stack
#include <utility>  // for move

#include "xgboost/base.h"        // for bst_node_t
#include "xgboost/tree_model.h"  // for RegTree

namespace xgboost::tree {
template <typename Base>
struct WalkTreeMixIn {
  /**
   * @brief Iterate through all nodes in this tree.
   *
   * @param Function that accepts a node index, and returns false when iteration should
   *        stop, otherwise returns true.
   */
  template <typename Func>
  void WalkTree(Func func) const {
    std::stack<bst_node_t> nodes;
    nodes.push(RegTree::kRoot);
    auto self = static_cast<Base const*>(this);
    while (!nodes.empty()) {
      auto nidx = nodes.top();
      nodes.pop();
      if (!func(nidx)) {
        return;
      }
      auto left = self->LeftChild(nidx);
      auto right = self->RightChild(nidx);
      if (left != RegTree::kInvalidNodeId) {
        nodes.push(left);
      }
      if (right != RegTree::kInvalidNodeId) {
        nodes.push(right);
      }
    }
  }
};

/**
 * @brief Tree view for scalar leaf.
 */
struct ScalarTreeView : public WalkTreeMixIn<ScalarTreeView> {
  static bst_node_t constexpr InvalidNodeId() { return RegTree::kInvalidNodeId; }
  static constexpr bst_node_t RootId() { return RegTree::kRoot; }

  RegTree::Node const* nodes;

  RTreeNodeStat const* stats;
  RegTree::CategoricalSplitMatrix cats;
  // The number of nodes
  bst_node_t n{0};

  [[nodiscard]] XGBOOST_DEVICE bool IsLeaf(bst_node_t nidx) const { return nodes[nidx].IsLeaf(); }
  [[nodiscard]] XGBOOST_DEVICE bst_node_t Parent(bst_node_t nidx) const {
    return nodes[nidx].Parent();
  }
  [[nodiscard]] XGBOOST_DEVICE bst_node_t LeftChild(bst_node_t nidx) const {
    return nodes[nidx].LeftChild();
  }
  [[nodiscard]] XGBOOST_DEVICE bst_node_t RightChild(bst_node_t nidx) const {
    return nodes[nidx].RightChild();
  }
  [[nodiscard]] XGBOOST_DEVICE bst_feature_t SplitIndex(bst_node_t nidx) const {
    return nodes[nidx].SplitIndex();
  }
  [[nodiscard]] XGBOOST_DEVICE bool IsDeleted(bst_node_t nidx) const {
    return nodes[nidx].IsDeleted();
  }
  [[nodiscard]] XGBOOST_DEVICE float SplitCond(bst_node_t nidx) const {
    return nodes[nidx].SplitCond();
  }
  [[nodiscard]] XGBOOST_DEVICE bool DefaultLeft(bst_node_t nidx) const {
    return nodes[nidx].DefaultLeft();
  }
  [[nodiscard]] XGBOOST_DEVICE bool IsLeftChild(bst_node_t nidx) const {
    return nodes[nidx].IsLeftChild();
  }
  [[nodiscard]] XGBOOST_DEVICE bst_node_t DefaultChild(bst_node_t nidx) const {
    return this->DefaultLeft(nidx) ? this->LeftChild(nidx) : this->RightChild(nidx);
  }
  [[nodiscard]] XGBOOST_DEVICE float LeafValue(bst_node_t nidx) const {
    return this->nodes[nidx].LeafValue();
  }

  [[nodiscard]] XGBOOST_DEVICE bst_node_t Size() const { return this->n; }
  [[nodiscard]] XGBOOST_DEVICE bool IsRoot(bst_node_t nidx) const {
    return this->nodes[nidx].IsRoot();
  }
  [[nodiscard]] RTreeNodeStat const& Stat(bst_node_t nidx) const { return stats[nidx]; }
  [[nodiscard]] FeatureType SplitType(bst_node_t nidx) const { return cats.split_type[nidx]; }

  [[nodiscard]] XGBOOST_DEVICE bool HasCategoricalSplit() const { return !cats.categories.empty(); }

  XGBOOST_DEVICE explicit ScalarTreeView(RegTree::Node const* nodes, RTreeNodeStat const* stats,
                                         RegTree::CategoricalSplitMatrix cats, bst_node_t n_nodes)
      : nodes{nodes}, stats{stats}, cats{std::move(cats)}, n{n_nodes} {}

  explicit ScalarTreeView(RegTree const* tree)
      : nodes{tree->GetNodes().data()},
        stats{tree->GetStats().data()},
        cats{tree->GetCategoriesMatrix()},
        n{tree->NumNodes()} {}

  /**
   * @brief Get the depth of a node.
   * @param nidx node id
   */
  [[nodiscard]] bst_node_t GetDepth(bst_node_t nidx) const {
    bst_node_t depth = 0;
    while (!nodes[nidx].IsRoot()) {
      ++depth;
      nidx = nodes[nidx].Parent();
    }
    return depth;
  }
};

/**
 * @brief A view to the @MultiTargetTree suitable for both host and device.
 */
struct MultiTargetTreeView : public WalkTreeMixIn<MultiTargetTreeView> {
  static bst_node_t constexpr InvalidNodeId() { return MultiTargetTree::InvalidNodeId(); }

  bst_node_t const* left;
  bst_node_t const* right;
  bst_node_t const* parent;

  bst_feature_t const* split_index;
  std::uint8_t const* default_left;
  float const* split_conds;

  RegTree::CategoricalSplitMatrix cats;

  // The number of nodes
  bst_node_t n{0};

  linalg::MatrixView<float const> weights;

  [[nodiscard]] XGBOOST_DEVICE bool IsLeaf(bst_node_t nidx) const {
    return left[nidx] == InvalidNodeId();
  }

  [[nodiscard]] XGBOOST_DEVICE bst_node_t LeftChild(bst_node_t nidx) const { return left[nidx]; }
  [[nodiscard]] XGBOOST_DEVICE bst_node_t RightChild(bst_node_t nidx) const { return right[nidx]; }
  [[nodiscard]] XGBOOST_DEVICE bst_feature_t SplitIndex(bst_node_t nidx) const {
    return split_index[nidx];
  }
  [[nodiscard]] XGBOOST_DEVICE float SplitCond(bst_node_t nidx) const { return split_conds[nidx]; }
  [[nodiscard]] XGBOOST_DEVICE bool DefaultLeft(bst_node_t nidx) const {
    return default_left[nidx];
  }
  [[nodiscard]] XGBOOST_DEVICE bst_node_t DefaultChild(bst_node_t nidx) const {
    return this->DefaultLeft(nidx) ? this->LeftChild(nidx) : this->RightChild(nidx);
  }
  [[nodiscard]] XGBOOST_DEVICE linalg::VectorView<float const> LeafValue(bst_node_t nidx) const {
    return this->weights.Slice(nidx, linalg::All());
  }

  [[nodiscard]] bst_target_t NumTargets() const { return this->weights.Shape(1); }
  [[nodiscard]] bst_node_t Size() const { return this->n; }

  explicit MultiTargetTreeView(Context const* ctx, RegTree const* tree);
};
}  // namespace xgboost::tree
