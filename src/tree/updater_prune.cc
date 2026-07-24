/**
 * Copyright 2014-2026, XGBoost Contributors
 * \file updater_prune.cc
 * \brief prune a tree given the statistics
 * \author Tianqi Chen
 */
#include <xgboost/tree_updater.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "../collective/communicator-inl.h"  // for GetRank, GetWorldSize
#include "../common/timer.h"
#include "./model_utils.h"  // for BroadcastTreeModel
#include "./param.h"
#include "xgboost/base.h"
#include "xgboost/gradient.h"  // for GradientContainer
#include "xgboost/json.h"

namespace xgboost::tree {
DMLC_REGISTRY_FILE_TAG(updater_prune);

/*! \brief pruner that prunes a tree after growing finishes */
class TreePruner : public TreeUpdater {
 public:
  explicit TreePruner(Context const* ctx) : TreeUpdater(ctx) { pruner_monitor_.Init("TreePruner"); }
  [[nodiscard]] char const* Name() const override { return "prune"; }
  // set training parameter
  void Configure(const Args&) override {}

  void LoadConfig(Json const&) override {}
  void SaveConfig(Json*) const override {}
  [[nodiscard]] bool CanModifyTree() const override { return true; }

  // update the tree, do pruning
  void Update(TrainParam const* param, GradientContainer*, DMatrix*,
              common::Span<HostDeviceVector<bst_node_t>>,
              std::vector<RegTree*> const& trees) override {
    pruner_monitor_.Start("PrunerUpdate");
    for (auto tree : trees) {
      this->DoPrune(param, tree);
    }
    this->Synchronize(trees);
    pruner_monitor_.Stop("PrunerUpdate");
  }

 private:
  // try to prune off current leaf
  bst_node_t TryPruneLeaf(TrainParam const* param, RegTree* p_tree, int nid, int depth,
                          int npruned) {
    auto& tree = *p_tree;
    CHECK(tree[nid].IsLeaf());
    if (tree[nid].IsRoot()) {
      return npruned;
    }
    bst_node_t pid = tree[nid].Parent();
    CHECK(!tree[pid].IsLeaf());
    RTreeNodeStat const& s = tree.Stat(pid);
    // Only prune when both child are leaf.
    auto left = tree[pid].LeftChild();
    auto right = tree[pid].RightChild();
    bool balanced = tree[left].IsLeaf() && right != RegTree::kInvalidNodeId && tree[right].IsLeaf();
    if (balanced && param->NeedPrune(s.loss_chg, depth)) {
      // need to be pruned
      tree.ChangeToLeaf(pid, param->learning_rate * s.base_weight);
      // tail recursion
      return this->TryPruneLeaf(param, p_tree, pid, depth - 1, npruned + 2);
    } else {
      return npruned;
    }
  }
  /*! \brief do pruning of a tree */
  void DoPrune(TrainParam const* param, RegTree* p_tree) {
    auto& tree = *p_tree;
    CHECK(!tree.IsMultiTarget()) << "Pruning" << MTNotImplemented();
    bst_node_t npruned = 0;
    for (int nid = 0; nid < tree.NumNodes(); ++nid) {
      if (tree[nid].IsLeaf() && !tree[nid].IsDeleted()) {
        npruned = this->TryPruneLeaf(param, p_tree, nid, tree.GetDepth(nid), npruned);
      }
    }
    LOG(INFO) << "tree pruning end, " << tree.NumExtraNodes() << " extra nodes, " << npruned
              << " pruned nodes, max_depth=" << tree.MaxDepth();
  }

  void Synchronize(std::vector<RegTree*> const& trees) {
    if (collective::GetWorldSize() == 1) {
      return;
    }

    auto rank = collective::GetRank();
    Json model{Array{}};
    if (rank == 0) {
      auto& tree_models = get<Array>(model);
      for (auto tree : trees) {
        Json tree_model{Object{}};
        tree->SaveModel(&tree_model);
        tree_models.emplace_back(std::move(tree_model));
      }
    }

    model = BroadcastTreeModel(ctx_, model);
    if (rank != 0) {
      auto const& tree_models = get<Array const>(model);
      CHECK_EQ(tree_models.size(), trees.size());
      for (std::size_t i = 0; i < trees.size(); ++i) {
        trees[i]->LoadModel(tree_models[i]);
      }
    }
  }

  common::Monitor pruner_monitor_;
};

XGBOOST_REGISTER_TREE_UPDATER(TreePruner, "prune")
    .describe("Pruner that prune the tree according to statistics.")
    .set_body([](Context const* ctx, ObjInfo const*) { return new TreePruner{ctx}; });
}  // namespace xgboost::tree
