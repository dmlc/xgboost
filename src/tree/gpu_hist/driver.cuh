/*!
 * Copyright 2020 by XGBoost Contributors
 */
#ifndef DRIVER_CUH_
#define DRIVER_CUH_
#include <xgboost/span.h>
#include <queue>
#include "../param.h"
#include "evaluate_splits.cuh"

namespace xgboost {
namespace tree {
struct ExpandEntry {
  int nid;
  int depth;
  DeviceSplitCandidate split;

  float base_weight { std::numeric_limits<float>::quiet_NaN() };
  float left_weight { std::numeric_limits<float>::quiet_NaN() };
  float right_weight { std::numeric_limits<float>::quiet_NaN() };

  ExpandEntry() = default;
  XGBOOST_DEVICE ExpandEntry(int nid, int depth, DeviceSplitCandidate split,
                             float base, float left, float right)
      : nid(nid), depth(depth), split(std::move(split)), base_weight{base},
        left_weight{left}, right_weight{right} {}
  bool IsValid(const TrainParam& param, int num_leaves) const {
    if (split.loss_chg <= kRtEps) return false;
    if (split.left_sum.GetHess() == 0 || split.right_sum.GetHess() == 0) {
      return false;
    }
    if (split.loss_chg < param.min_split_loss) {
      return false;
    }
    if (param.max_depth > 0 && depth == param.max_depth) {
      return false;
    }
    if (param.max_leaves > 0 && num_leaves == param.max_leaves) {
      return false;
    }
    return true;
  }

  static bool ChildIsValid(const TrainParam& param, int depth, int num_leaves) {
    if (param.max_depth > 0 && depth >= param.max_depth) return false;
    if (param.max_leaves > 0 && num_leaves >= param.max_leaves) return false;
    return true;
  }

  friend std::ostream& operator<<(std::ostream& os, const ExpandEntry& e) {
    os << "ExpandEntry: \n";
    os << "nidx: " << e.nid << "\n";
    os << "depth: " << e.depth << "\n";
    os << "loss: " << e.split.loss_chg << "\n";
    os << "left_sum: " << e.split.left_sum << "\n";
    os << "right_sum: " << e.split.right_sum << "\n";
    return os;
  }
};

inline bool DepthWise(const ExpandEntry& lhs, const ExpandEntry& rhs) {
  return lhs.depth > rhs.depth;  // favor small depth
}

inline bool LossGuide(const ExpandEntry& lhs, const ExpandEntry& rhs) {
  if (lhs.split.loss_chg == rhs.split.loss_chg) {
    return lhs.nid > rhs.nid;  // favor small timestamp
  } else {
    return lhs.split.loss_chg < rhs.split.loss_chg;  // favor large loss_chg
  }
}

// Drives execution of tree building on device
class Driver {
  using ExpandQueue =
      std::priority_queue<ExpandEntry, std::vector<ExpandEntry>,
                          std::function<bool(ExpandEntry, ExpandEntry)>>;

 public:
  explicit Driver(TrainParam::TreeGrowPolicy policy)
      : policy_(policy),
        queue_(policy == TrainParam::kDepthWise ? DepthWise : LossGuide) {}
  template <typename EntryIterT>
  void Push(EntryIterT begin,EntryIterT end) {
    for (auto it = begin; it != end; ++it) {
      const ExpandEntry& e = *it;
      if (e.split.loss_chg > kRtEps) {
        queue_.push(e);
      }
    }
  }
  void Push(const std::vector<ExpandEntry> &entries) {
    this->Push(entries.begin(), entries.end());
  }
  // Return the set of nodes to be expanded
  // This set has no dependencies between entries so they may be expanded in
  // parallel or asynchronously
  std::vector<ExpandEntry> Pop() {
    if (queue_.empty()) return {};
    // Return a single entry for loss guided mode
    if (policy_ == TrainParam::kLossGuide) {
      ExpandEntry e = queue_.top();
      queue_.pop();
      return {e};
    }
    // Return nodes on same level for depth wise
    std::vector<ExpandEntry> result;
    ExpandEntry e = queue_.top();
    int level = e.depth;
    while (e.depth == level && !queue_.empty()) {
      queue_.pop();
      result.emplace_back(e);
      if (!queue_.empty()) {
        e = queue_.top();
      }
    }
    return result;
  }

 private:
  TrainParam::TreeGrowPolicy policy_;
  ExpandQueue queue_;
};
}  // namespace tree
}  // namespace xgboost

#endif  // DRIVER_CUH_
