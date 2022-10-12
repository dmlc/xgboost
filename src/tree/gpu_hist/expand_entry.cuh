/*!
 * Copyright 2020 by XGBoost Contributors
 */
#ifndef EXPAND_ENTRY_CUH_
#define EXPAND_ENTRY_CUH_
#include <xgboost/span.h>

#include "../param.h"
#include "../updater_gpu_common.cuh"

namespace xgboost {
namespace tree {

struct GPUExpandEntry {
  int nid;
  int depth;
  DeviceSplitCandidate split;

  float base_weight { std::numeric_limits<float>::quiet_NaN() };
  float left_weight { std::numeric_limits<float>::quiet_NaN() };
  float right_weight { std::numeric_limits<float>::quiet_NaN() };

  GPUExpandEntry() = default;
  XGBOOST_DEVICE GPUExpandEntry(int nid, int depth, DeviceSplitCandidate split,
                             float base, float left, float right)
      : nid(nid), depth(depth), split(std::move(split)), base_weight{base},
        left_weight{left}, right_weight{right} {}
  bool IsValid(const TrainParam& param, int num_leaves) const {
    if (split.loss_chg <= kRtEps) return false;
    if (split.left_sum.GetQuantisedHess() == 0 || split.right_sum.GetQuantisedHess() == 0) {
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

  bst_float GetLossChange() const {
    return split.loss_chg;
  }

  int GetNodeId() const {
    return nid;
  }

  int GetDepth() const {
    return depth;
  }

  friend std::ostream& operator<<(std::ostream& os, const GPUExpandEntry& e) {
    os << "GPUExpandEntry: \n";
    os << "nidx: " << e.nid << "\n";
    os << "depth: " << e.depth << "\n";
    os << "loss: " << e.split.loss_chg << "\n";
    os << "left_sum: " << e.split.left_sum << "\n";
    os << "right_sum: " << e.split.right_sum << "\n";
    return os;
  }
};

}  // namespace tree
}  // namespace xgboost

#endif  // EXPAND_ENTRY_CUH_
