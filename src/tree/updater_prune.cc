/*!
 * Copyright 2014 by Contributors
 * \file updater_prune.cc
 * \brief prune a tree given the statistics
 * \author Tianqi Chen
 */

#include <xgboost/tree_updater.h>
#include <string>
#include <memory>
#include "./param.h"
#include "../common/sync.h"
#include "../common/io.h"

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_prune);

/*! \brief pruner that prunes a tree after growing finishes */
class TreePruner: public TreeUpdater {
 public:
  TreePruner() {
    syncher_.reset(TreeUpdater::Create("sync"));
  }
  // set training parameter
  void Init(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
    syncher_->Init(args);
  }
  // update the tree, do pruning
  void Update(HostDeviceVector<GradientPair> *gpair,
              DMatrix *p_fmat,
              const std::vector<RegTree*> &trees) override {
    // rescale learning rate according to size of trees
    float lr = param_.learning_rate;
    param_.learning_rate = lr / trees.size();
    for (auto tree : trees) {
      this->DoPrune(*tree);
    }
    param_.learning_rate = lr;
    syncher_->Update(gpair, p_fmat, trees);
  }

 private:
  // try to prune off current leaf
  inline int TryPruneLeaf(RegTree &tree, int nid, int depth, int npruned) { // NOLINT(*)
    if (tree[nid].IsRoot()) return npruned;
    int pid = tree[nid].Parent();
    RegTree::NodeStat &s = tree.Stat(pid);
    ++s.leaf_child_cnt;
    if (s.leaf_child_cnt >= 2 && param_.NeedPrune(s.loss_chg, depth - 1)) {
      // need to be pruned
      tree.ChangeToLeaf(pid, param_.learning_rate * s.base_weight);
      // tail recursion
      return this->TryPruneLeaf(tree, pid, depth - 1, npruned + 2);
    } else {
      return npruned;
    }
  }
  /*! \brief do pruning of a tree */
  inline void DoPrune(RegTree &tree) { // NOLINT(*)
    int npruned = 0;
    // initialize auxiliary statistics
    for (int nid = 0; nid < tree.param.num_nodes; ++nid) {
      tree.Stat(nid).leaf_child_cnt = 0;
    }
    for (int nid = 0; nid < tree.param.num_nodes; ++nid) {
      if (tree[nid].IsLeaf()) {
        npruned = this->TryPruneLeaf(tree, nid, tree.GetDepth(nid), npruned);
      }
    }
    if (!param_.silent) {
      LOG(INFO) << "tree pruning end, " << tree.param.num_roots << " roots, "
                << tree.NumExtraNodes() << " extra nodes, " << npruned
                << " pruned nodes, max_depth=" << tree.MaxDepth();
    }
  }

 private:
  // synchronizer
  std::unique_ptr<TreeUpdater> syncher_;
  // training parameter
  TrainParam param_;
};

XGBOOST_REGISTER_TREE_UPDATER(TreePruner, "prune")
.describe("Pruner that prune the tree according to statistics.")
.set_body([]() {
    return new TreePruner();
  });
}  // namespace tree
}  // namespace xgboost
