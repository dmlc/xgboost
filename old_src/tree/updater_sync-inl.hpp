/*!
 * Copyright 2014 by Contributors
 * \file updater_sync-inl.hpp
 * \brief synchronize the tree in all distributed nodes
 * \author Tianqi Chen
 */
#ifndef XGBOOST_TREE_UPDATER_SYNC_INL_HPP_
#define XGBOOST_TREE_UPDATER_SYNC_INL_HPP_

#include <vector>
#include <string>
#include <limits>
#include "../sync/sync.h"
#include "./updater.h"

namespace xgboost {
namespace tree {
/*!
 * \brief syncher that synchronize the tree in all distributed nodes
 * can implement various strategies, so far it is always set to node 0's tree
 */
class TreeSyncher: public IUpdater {
 public:
  virtual ~TreeSyncher(void) {}
  virtual void SetParam(const char *name, const char *val) {
  }
  // update the tree, do pruning
  virtual void Update(const std::vector<bst_gpair> &gpair,
                      IFMatrix *p_fmat,
                      const BoosterInfo &info,
                      const std::vector<RegTree*> &trees) {
    this->SyncTrees(trees);
  }

 private:
  // synchronize the trees in different nodes, take tree from rank 0
  inline void SyncTrees(const std::vector<RegTree *> &trees) {
    if (rabit::GetWorldSize() == 1) return;
    std::string s_model;
    utils::MemoryBufferStream fs(&s_model);
    int rank = rabit::GetRank();
    if (rank == 0) {
      for (size_t i = 0; i < trees.size(); ++i) {
        trees[i]->SaveModel(fs);
      }
    }
    fs.Seek(0);
    rabit::Broadcast(&s_model, 0);
    for (size_t i = 0; i < trees.size(); ++i) {
      trees[i]->LoadModel(fs);
    }
  }
};
}  // namespace tree
}  // namespace xgboost
#endif  // XGBOOST_TREE_UPDATER_SYNC_INL_HPP_
