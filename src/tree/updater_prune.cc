/**
 * Copyright 2014-2023 by XGBoost Contributors
 * \file updater_prune.cc
 * \brief prune a tree given the statistics
 * \author Tianqi Chen
 */
#include <xgboost/tree_updater.h>

#include <memory>

#include "../common/timer.h"
#include "./param.h"
#include "xgboost/base.h"
#include "xgboost/json.h"
namespace xgboost::tree {
DMLC_REGISTRY_FILE_TAG(updater_prune);

/*! \brief pruner that prunes a tree after growing finishes */
class TreePruner : public TreeUpdater {
 public:
  explicit TreePruner(Context const* ctx, ObjInfo const* task) : TreeUpdater(ctx) {
    syncher_.reset(TreeUpdater::Create("sync", ctx_, task));
    pruner_monitor_.Init("TreePruner");
  }
  [[nodiscard]] char const* Name() const override { return "prune"; }
  // set training parameter
  void Configure(const Args& args) override { syncher_->Configure(args); }

  void LoadConfig(Json const&) override {}
  void SaveConfig(Json*) const override {}
  [[nodiscard]] bool CanModifyTree() const override { return true; }

  // update the tree, do pruning
  void Update(TrainParam const* param, HostDeviceVector<GradientPair>* gpair, DMatrix* p_fmat,
              common::Span<HostDeviceVector<bst_node_t>> out_position,
              const std::vector<RegTree*>& trees) override {
    pruner_monitor_.Start("PrunerUpdate");
    for (auto tree : trees) {
      this->DoPrune(param, tree);
    }
    syncher_->Update(param, gpair, p_fmat, out_position, trees);
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
    RTreeNodeStat const &s = tree.Stat(pid);
    // Only prune when both child are leaf.
    auto left = tree[pid].LeftChild();
    auto right = tree[pid].RightChild();
    bool balanced = tree[left].IsLeaf() &&
                    right != RegTree::kInvalidNodeId && tree[right].IsLeaf();
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
    bst_node_t npruned = 0;
    for (int nid = 0; nid < tree.NumNodes(); ++nid) {
      if (tree[nid].IsLeaf() && !tree[nid].IsDeleted()) {
        npruned = this->TryPruneLeaf(param, p_tree, nid, tree.GetDepth(nid), npruned);
      }
    }
    LOG(INFO) << "tree pruning end, "
              << tree.NumExtraNodes() << " extra nodes, " << npruned
              << " pruned nodes, max_depth=" << tree.MaxDepth();
  }

 private:
  // synchronizer
  std::unique_ptr<TreeUpdater> syncher_;
  common::Monitor pruner_monitor_;
};

XGBOOST_REGISTER_TREE_UPDATER(TreePruner, "prune")
    .describe("Pruner that prune the tree according to statistics.")
    .set_body([](Context const* ctx, ObjInfo const* task) {
      return new TreePruner{ctx, task};
    });
}  // namespace xgboost::tree
