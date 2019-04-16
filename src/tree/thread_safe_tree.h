/*!
 * Copyright 2019 by Contributors
 * \file thread_safe_tree.h
 * \brief Utility for fast histogram aggregation
 * \author Philip Cho, Tianqi Chen
 */
#ifndef XGBOOST_TREE_THREAD_SAFE_TREE_H_
#define XGBOOST_TREE_THREAD_SAFE_TREE_H_

#include <vector>
#include "xgboost/tree_model.h"

namespace xgboost {
namespace tree {

class RegTreeThreadSafe {
 public:
RegTreeThreadSafe(RegTree* p_tree, const std::vector<typename QuantileHistMaker::NodeEntry>& snode,
    const TrainParam& param): p_tree_(*p_tree), snode_(snode.size()), param_(param) { }

~RegTreeThreadSafe() {
  for (size_t i = 0; i < snode_.size(); ++i) {
    delete snode_[i];
    snode_[i] = nullptr;
  }
}

const RegTree& Get() const {
  return p_tree_;
}

const TreeParam& Param() const {
  return p_tree_.param;
}

RegTree::Node operator[](size_t idx) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return p_tree_[idx];
}

void SetLeaf(bst_float value, int32_t nid) {
  std::lock_guard<std::mutex> lock(mutex_);
  p_tree_[nid].SetLeaf(value);
}


RegTree::Node ExpandNode(int nid, unsigned split_index, bst_float split_value,
                bool default_left, bst_float base_weight,
                bst_float left_leaf_weight, bst_float right_leaf_weight,
                bst_float loss_change, float sum_hess) {
  std::lock_guard<std::mutex> lock(mutex_);
  p_tree_.ExpandNode(nid, split_index, split_value, default_left, base_weight,
                left_leaf_weight, right_leaf_weight, loss_change, sum_hess);
  return p_tree_[nid];
}

size_t GetDepth(size_t nid) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return p_tree_.GetDepth(nid);
}

size_t NumNodes() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return Param().num_nodes;
}

QuantileHistMaker::NodeEntry& Snode(size_t nid) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (nid >= snode_.size())
    resize(nid+1);

  return *(snode_[nid]);
}

const QuantileHistMaker::NodeEntry& Snode(size_t nid) const {
  std::lock_guard<std::mutex> lock(mutex_);

  if (nid >= snode_.size())
    resize(nid+1);

  return *(snode_[nid]);
}

void ResizeSnode(const TrainParam& param) {
  std::lock_guard<std::mutex> lock(mutex_);
  int n_nodes = Param().num_nodes;
  resize(n_nodes);
}

 protected:
  void resize(int n_nodes) const {
    int prev_size = snode_.size();
    snode_.resize(n_nodes, nullptr);

    if (prev_size < n_nodes) {
      for (int i = prev_size; i < n_nodes; ++i) {
        if (snode_[i] != nullptr) delete snode_[i];
        snode_[i] = new typename QuantileHistMaker::NodeEntry(param_);
      }
    }
  }

  mutable std::mutex mutex_;
  RegTree& p_tree_;
  mutable std::vector<typename QuantileHistMaker::NodeEntry*> snode_;
  const TrainParam& param_;
};

}  // namespace tree
}  // namespace xgboost

#endif  // XGBOOST_TREE_THREAD_SAFE_TREE_H_
