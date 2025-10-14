/**
 * Copyright 2025, XGBoost Contributors
 *
 * The file provides views for two tree models. We hope to eventually unify them, but the
 * original scalar tree `Node` struct is used extensively in the codebase.
 */
#pragma once
#include <algorithm>  // for max
#include <cstdint>    // for uint8_t
#include <stack>      // for stack
#include <utility>    // for move

#include "../common/type.h"      // for GetValueT
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
  template <typename Fn>
  void WalkTree(Fn&& func) const {
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

  /**
   * @brief Get the depth of a node.
   * @param nidx node id
   */
  [[nodiscard]] bst_node_t GetDepth(bst_node_t nidx) const {
    bst_node_t depth = 0;
    auto self = static_cast<Base const*>(this);
    while (!self->IsRoot(nidx)) {
      ++depth;
      nidx = self->Parent(nidx);
    }
    return depth;
  }

  [[nodiscard]] bst_node_t MaxDepth(bst_node_t nidx) const {
    auto self = static_cast<Base const*>(this);
    if (self->IsLeaf(nidx)) {
      return 0;
    }
    return std::max(this->MaxDepth(self->LeftChild(nidx)) + 1,
                    this->MaxDepth(self->RightChild(nidx)) + 1);
  }
  [[nodiscard]] bst_node_t MaxDepth() const { return this->MaxDepth(RegTree::kRoot); }
};

struct CategoriesMixIn {
  RegTree::CategoricalSplitMatrix cats;

  [[nodiscard]] XGBOOST_DEVICE bool HasCategoricalSplit() const { return !cats.categories.empty(); }
  [[nodiscard]] XGBOOST_DEVICE RegTree::CategoricalSplitMatrix GetCategoriesMatrix() const {
    return cats;
  }
  /**
   * @brief Get the bit storage for categories
   */
  [[nodiscard]] common::Span<uint32_t const> NodeCats(bst_node_t nidx) const {
    auto node_ptr = this->GetCategoriesMatrix().node_ptr;
    auto categories = this->GetCategoriesMatrix().categories;
    auto segment = node_ptr[nidx];
    auto node_cats = categories.subspan(segment.beg, segment.size);
    return node_cats;
  }
  [[nodiscard]] FeatureType SplitType(bst_node_t nidx) const { return cats.split_type[nidx]; }
};

/**
 * @brief Tree view for scalar leaf.
 */
struct ScalarTreeView : public WalkTreeMixIn<ScalarTreeView>, public CategoriesMixIn {
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

  [[nodiscard]] bst_target_t NumTargets() const { return 1; }
  [[nodiscard]] XGBOOST_DEVICE bst_node_t Size() const { return this->n; }
  [[nodiscard]] XGBOOST_DEVICE bool IsRoot(bst_node_t nidx) const {
    return this->nodes[nidx].IsRoot();
  }

  [[nodiscard]] RTreeNodeStat const& Stat(bst_node_t nidx) const { return stats[nidx]; }
  [[nodiscard]] auto SumHess(bst_node_t nidx) const { return stats[nidx].sum_hess; }
  [[nodiscard]] auto LossChg(bst_node_t nidx) const { return stats[nidx].loss_chg; }

  XGBOOST_DEVICE explicit ScalarTreeView(RegTree::Node const* nodes, RTreeNodeStat const* stats,
                                         RegTree::CategoricalSplitMatrix cats, bst_node_t n_nodes)
      : nodes{nodes}, stats{stats}, cats{std::move(cats)}, n{n_nodes} {}

  /** @brief Create a device view, not implemented yet. */
  explicit ScalarTreeView(Context const* ctx, RegTree const* tree);
  /** @brief Create a host view */
  explicit ScalarTreeView(RegTree const* tree)
      : CategoriesMixIn{tree->GetCategoriesMatrix()},
        nodes{tree->GetNodes().data()},
        stats{tree->GetStats().data()},
        n{tree->NumNodes()} {
    CHECK(!tree->IsMultiTarget());
  }
};

/**
 * @brief A view to the @ref MultiTargetTree suitable for both host and device.
 */
struct MultiTargetTreeView : public WalkTreeMixIn<MultiTargetTreeView>, public CategoriesMixIn {
  static bst_node_t constexpr InvalidNodeId() { return MultiTargetTree::InvalidNodeId(); }

  bst_node_t const* left;
  bst_node_t const* right;
  bst_node_t const* parent;

  bst_feature_t const* split_index;
  std::uint8_t const* default_left;
  float const* split_conds;

  // The number of nodes
  bst_node_t n{0};

  linalg::MatrixView<float const> weights;

  [[nodiscard]] XGBOOST_DEVICE bool IsLeaf(bst_node_t nidx) const {
    return left[nidx] == InvalidNodeId();
  }

  [[nodiscard]] XGBOOST_DEVICE bst_node_t Parent(bst_node_t nidx) const { return parent[nidx]; }
  [[nodiscard]] XGBOOST_DEVICE bst_node_t LeftChild(bst_node_t nidx) const { return left[nidx]; }
  [[nodiscard]] XGBOOST_DEVICE bst_node_t RightChild(bst_node_t nidx) const { return right[nidx]; }

  [[nodiscard]] bool IsLeftChild(bst_node_t nidx) const {
    auto p = this->Parent(nidx);
    return nidx == this->LeftChild(p);
  }
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
  [[nodiscard]] XGBOOST_DEVICE bool IsRoot(bst_node_t nidx) const { return nidx == RegTree::kRoot; }

  [[nodiscard]] auto SumHess(bst_node_t) const {
    LOG(FATAL) << "Tree statistic " << MTNotImplemented();
    return linalg::MakeVec<float>(nullptr, 0);
  }
  [[nodiscard]] auto LossChg(bst_node_t) const {
    LOG(FATAL) << "Tree statistic " << MTNotImplemented();
    return 0.0f;
  }
  /** @brief Create a device view */
  explicit MultiTargetTreeView(Context const* ctx, RegTree const* tree);
  /** @brief Create a host view */
  explicit MultiTargetTreeView(RegTree const* tree);
};

template <typename Fn, typename... Tree>
void WalkTree(RegTree const& tree, Fn&& fn, Tree const&... trees) {
  if (tree.IsMultiTarget()) {
    auto mt_tree = tree.HostMtView();
    mt_tree.WalkTree([&](bst_node_t nidx) { return fn(mt_tree, trees.HostMtView()..., nidx); });
  } else {
    auto sc_tree = tree.HostScView();
    sc_tree.WalkTree([&](bst_node_t nidx) { return fn(sc_tree, trees.HostScView()..., nidx); });
  }
}

template <typename TreeView>
[[nodiscard]] bool constexpr IsScalarTree() {
  return std::is_same_v<common::GetValueT<TreeView>, ScalarTreeView>;
}

template <typename TreeView>
[[nodiscard]] bool constexpr IsScalarTree(TreeView const&) {
  return IsScalarTree<TreeView>();
}
}  // namespace xgboost::tree
