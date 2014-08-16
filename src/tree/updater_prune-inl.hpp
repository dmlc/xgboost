#ifndef XGBOOST_TREE_UPDATER_PRUNE_INL_HPP_
#define XGBOOST_TREE_UPDATER_PRUNE_INL_HPP_
/*!
 * \file updater_prune-inl.hpp
 * \brief prune a tree given the statistics 
 * \author Tianqi Chen
 */
#include <vector>
#include "./param.h"
#include "./updater.h"

namespace xgboost {
namespace tree {
/*! \brief pruner that prunes a tree after growing finishs */
template<typename FMatrix>
class TreePruner: public IUpdater<FMatrix> {
 public:
  virtual ~TreePruner(void) {}
  // set training parameter
  virtual void SetParam(const char *name, const char *val) {
    param.SetParam(name, val);
  }
  // update the tree, do pruning
  virtual void Update(const std::vector<bst_gpair> &gpair,
                      const FMatrix &fmat,
                      const std::vector<unsigned> &root_index,
                      const std::vector<RegTree*> &trees) {
    for (size_t i = 0; i < trees.size(); ++i) {
      this->DoPrune(*trees[i]);
    }
  }

 private:
  // try to prune off current leaf
  inline void TryPruneLeaf(RegTree &tree, int nid, int depth) {
    if (tree[nid].is_root()) return;
    int pid = tree[nid].parent();
    RegTree::NodeStat &s = tree.stat(pid);
    ++s.leaf_child_cnt;

    if (s.leaf_child_cnt >= 2 && param.need_prune(s.loss_chg, depth - 1)) {
      // need to be pruned
      tree.ChangeToLeaf(pid, param.learning_rate * s.base_weight);
      // tail recursion
      this->TryPruneLeaf(tree, pid, depth - 1);
    }
  }
  /*! \brief do prunning of a tree */
  inline void DoPrune(RegTree &tree) {
    // initialize auxiliary statistics
    for (int nid = 0; nid < tree.param.num_nodes; ++nid) {
      tree.stat(nid).leaf_child_cnt = 0;
    }
    for (int nid = 0; nid < tree.param.num_nodes; ++nid) {
      if (tree[nid].is_leaf()) {
        this->TryPruneLeaf(tree, nid, tree.GetDepth(nid));
      }
    }
  }

 private:
  // training parameter
  TrainParam param;
};

}  // namespace tree
}  // namespace xgboost
#endif  // XGBOOST_TREE_UPDATER_PRUNE_INL_HPP_
