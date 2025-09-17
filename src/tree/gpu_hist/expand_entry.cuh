/**
 * Copyright 2020-2024, XGBoost Contributors
 */
#ifndef EXPAND_ENTRY_CUH_
#define EXPAND_ENTRY_CUH_

#include <limits>   // for numeric_limits
#include <utility>  // for move

#include "../param.h"                 // for TrainParam
#include "../updater_gpu_common.cuh"  // for DeviceSplitCandidate
#include "xgboost/base.h"             // for bst_node_t

namespace xgboost::tree {
struct GPUExpandEntry {
  bst_node_t nid;
  bst_node_t depth;
  DeviceSplitCandidate split;

  float base_weight{std::numeric_limits<float>::quiet_NaN()};
  float left_weight{std::numeric_limits<float>::quiet_NaN()};
  float right_weight{std::numeric_limits<float>::quiet_NaN()};

  GPUExpandEntry() = default;
  XGBOOST_DEVICE GPUExpandEntry(bst_node_t nid, bst_node_t depth, DeviceSplitCandidate split,
                                float base, float left, float right)
      : nid(nid),
        depth(depth),
        split(std::move(split)),
        base_weight{base},
        left_weight{left},
        right_weight{right} {}
  [[nodiscard]] bool IsValid(TrainParam const& param, bst_node_t num_leaves) const {
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

  [[nodiscard]] float GetLossChange() const { return split.loss_chg; }

  [[nodiscard]] bst_node_t GetNodeId() const { return nid; }

  [[nodiscard]] bst_node_t GetDepth() const { return depth; }

  friend std::ostream& operator<<(std::ostream& os, const GPUExpandEntry& e) {
    os << "GPUExpandEntry: \n";
    os << "nidx: " << e.nid << "\n";
    os << "depth: " << e.depth << "\n";
    os << "loss: " << e.split.loss_chg << "\n";
    os << "left_sum: " << e.split.left_sum << "\n";
    os << "right_sum: " << e.split.right_sum << "\n";
    return os;
  }

  void Save(Json* p_out) const {
    auto& out = *p_out;

    out["nid"] = Integer{this->nid};
    out["depth"] = Integer{this->depth};
    // GPU specific
    out["base_weight"] = this->base_weight;
    out["left_weight"] = this->left_weight;
    out["right_weight"] = this->right_weight;

    /**
     * Handle split
     */
    out["split"] = Object{};
    auto& split = out["split"];
    split["loss_chg"] = this->split.loss_chg;
    split["sindex"] = Integer{this->split.findex};
    split["split_value"] = this->split.fvalue;

    // cat
    split["thresh"] = Integer{this->split.thresh};
    split["is_cat"] = Boolean{this->split.is_cat};
    /**
     * Gradients
     */
    auto save = [&](std::string const& name, GradientPairInt64 const& sum) {
      out[name] = I64Array{2};
      auto& array = get<I64Array>(out[name]);
      array[0] = sum.GetQuantisedGrad();
      array[1] = sum.GetQuantisedHess();
    };
    save("left_sum", this->split.left_sum);
    save("right_sum", this->split.right_sum);
  }

  void Load(Json const& in) {
    this->nid = get<Integer const>(in["nid"]);
    this->depth = get<Integer const>(in["depth"]);
    // GPU specific
    this->base_weight = get<Number const>(in["base_weight"]);
    this->left_weight = get<Number const>(in["left_weight"]);
    this->right_weight = get<Number const>(in["right_weight"]);

    /**
     * Handle split
     */
    auto const& split = in["split"];
    this->split.loss_chg = get<Number const>(split["loss_chg"]);
    this->split.findex = get<Integer const>(split["sindex"]);
    this->split.fvalue = get<Number const>(split["split_value"]);
    // cat
    this->split.thresh = get<Integer const>(split["thresh"]);
    this->split.is_cat = get<Boolean const>(split["is_cat"]);
    /**
     * Gradients
     */
    auto const& left_sum = get<I64Array const>(in["left_sum"]);
    this->split.left_sum = GradientPairInt64{left_sum[0], left_sum[1]};
    auto const& right_sum = get<I64Array const>(in["right_sum"]);
    this->split.right_sum = GradientPairInt64{right_sum[0], right_sum[1]};
  }
};
}  // namespace xgboost::tree

#endif  // EXPAND_ENTRY_CUH_
