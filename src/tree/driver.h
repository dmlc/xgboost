/*!
 * Copyright 2021 by XGBoost Contributors
 */
#ifndef XGBOOST_TREE_DRIVER_H_
#define XGBOOST_TREE_DRIVER_H_
#include <xgboost/span.h>
#include <queue>
#include <vector>
#include "./param.h"

namespace xgboost {
namespace tree {

template <typename ExpandEntryT>
inline bool DepthWise(const ExpandEntryT& lhs, const ExpandEntryT& rhs) {
  return lhs.GetNodeId() > rhs.GetNodeId();  // favor small depth
}

template <typename ExpandEntryT>
inline bool LossGuide(const ExpandEntryT& lhs, const ExpandEntryT& rhs) {
  if (lhs.GetLossChange() == rhs.GetLossChange()) {
    return lhs.GetNodeId() > rhs.GetNodeId();  // favor small timestamp
  } else {
    return lhs.GetLossChange() < rhs.GetLossChange();  // favor large loss_chg
  }
}

// Drives execution of tree building on device
template <typename ExpandEntryT>
class Driver {
  using ExpandQueue =
      std::priority_queue<ExpandEntryT, std::vector<ExpandEntryT>,
                          std::function<bool(ExpandEntryT, ExpandEntryT)>>;

 public:
  explicit Driver(TrainParam param, std::size_t max_node_batch_size = 256)
      : param_(param),
        max_node_batch_size_(max_node_batch_size),
        queue_(param.grow_policy == TrainParam::kDepthWise ? DepthWise<ExpandEntryT>
                                                           : LossGuide<ExpandEntryT>) {}
  template <typename EntryIterT>
  void Push(EntryIterT begin, EntryIterT end) {
    for (auto it = begin; it != end; ++it) {
      const ExpandEntryT& e = *it;
      if (e.split.loss_chg > kRtEps) {
        queue_.push(e);
      }
    }
  }
  void Push(const std::vector<ExpandEntryT> &entries) {
    this->Push(entries.begin(), entries.end());
  }
  void Push(ExpandEntryT const& e) { queue_.push(e); }

  bool IsEmpty() {
    return queue_.empty();
  }

  // Can a child of this entry still be expanded?
  // can be used to avoid extra work
  bool IsChildValid(ExpandEntryT const& parent_entry) {
    if (param_.max_depth > 0 && parent_entry.depth + 1 >= param_.max_depth) return false;
    if (param_.max_leaves > 0 && num_leaves_ >= param_.max_leaves) return false;
    return true;
  }

  // Return the set of nodes to be expanded
  // This set has no dependencies between entries so they may be expanded in
  // parallel or asynchronously
  std::vector<ExpandEntryT> Pop() {
    if (queue_.empty()) return {};
    // Return a single entry for loss guided mode
    if (param_.grow_policy == TrainParam::kLossGuide) {
      ExpandEntryT e = queue_.top();
      queue_.pop();

      if (e.IsValid(param_, num_leaves_)) {
        num_leaves_++;
        return {e};
      } else {
        return {};
      }
    }
    // Return nodes on same level for depth wise
    std::vector<ExpandEntryT> result;
    ExpandEntryT e = queue_.top();
    int level = e.depth;
    while (e.depth == level && !queue_.empty() && result.size() < max_node_batch_size_) {
      queue_.pop();
      if (e.IsValid(param_, num_leaves_)) {
        num_leaves_++;
        result.emplace_back(e);
      }

      if (!queue_.empty()) {
        e = queue_.top();
      }
    }
    return result;
  }

 private:
  TrainParam param_;
  bst_node_t num_leaves_ = 1;
  std::size_t max_node_batch_size_;
  ExpandQueue queue_;
};
}  // namespace tree
}  // namespace xgboost

#endif  // XGBOOST_TREE_DRIVER_H_
