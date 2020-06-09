/*!
 * Copyright 2014-2020 by Contributors
 * \file updater_prune.cc
 * \brief prune a tree given the statistics
 * \author Tianqi Chen
 */
#include <rabit/rabit.h>
#include <xgboost/tree_updater.h>

#include <string>
#include <memory>

#include "xgboost/base.h"
#include "xgboost/json.h"
#include "./param.h"
#include "../common/io.h"
#include "../common/timer.h"
namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_prune);

/*! \brief pruner that prunes a tree after growing finishes */
class TreePruner: public TreeUpdater {
 public:
  TreePruner() {
    syncher_.reset(TreeUpdater::Create("sync", tparam_));
    pruner_monitor_.Init("TreePruner");
  }
  char const* Name() const override {
    return "prune";
  }

  // set training parameter
  void Configure(const Args& args) override {
    param_.UpdateAllowUnknown(args);
    syncher_->Configure(args);
  }

  void LoadConfig(Json const& in) override {
    auto const& config = get<Object const>(in);
    FromJson(config.at("train_param"), &this->param_);
  }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["train_param"] = ToJson(param_);
  }
  bool CanModifyTree() const override {
    return true;
  }

  // update the tree, do pruning
  void Update(HostDeviceVector<GradientPair> *gpair,
              DMatrix *p_fmat,
              const std::vector<RegTree*> &trees) override {
    pruner_monitor_.Start("PrunerUpdate");
    // rescale learning rate according to size of trees
    float lr = param_.learning_rate;
    param_.learning_rate = lr / trees.size();
    for (auto tree : trees) {
      this->DoPrune(tree);
    }
    param_.learning_rate = lr;
    syncher_->Update(gpair, p_fmat, trees);
    pruner_monitor_.Stop("PrunerUpdate");
  }

 private:
  // try to prune off current leaf
  bst_node_t TryPruneLeaf(RegTree &tree, int nid, int depth, int npruned) { // NOLINT(*)
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
    if (balanced && param_.NeedPrune(s.loss_chg, depth)) {
      // need to be pruned
      tree.ChangeToLeaf(pid, param_.learning_rate * s.base_weight);
      // tail recursion
      return this->TryPruneLeaf(tree, pid, depth - 1, npruned + 2);
    } else {
      return npruned;
    }
  }
  /*! \brief do pruning of a tree */
  void DoPrune(RegTree* p_tree) {
    auto& tree = *p_tree;
    bst_node_t npruned = 0;
    for (int nid = 0; nid < tree.param.num_nodes; ++nid) {
      if (tree[nid].IsLeaf() && !tree[nid].IsDeleted()) {
        npruned = this->TryPruneLeaf(tree, nid, tree.GetDepth(nid), npruned);
      }
    }
    LOG(INFO) << "tree pruning end, "
              << tree.NumExtraNodes() << " extra nodes, " << npruned
              << " pruned nodes, max_depth=" << tree.MaxDepth();
  }

 private:
  // synchronizer
  std::unique_ptr<TreeUpdater> syncher_;
  // training parameter
  TrainParam param_;
  common::Monitor pruner_monitor_;
};

XGBOOST_REGISTER_TREE_UPDATER(TreePruner, "prune")
.describe("Pruner that prune the tree according to statistics.")
.set_body([]() {
    return new TreePruner();
  });
}  // namespace tree
}  // namespace xgboost
