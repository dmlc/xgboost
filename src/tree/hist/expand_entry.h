/**
 * Copyright 2021-2023 XGBoost contributors
 */
#ifndef XGBOOST_TREE_HIST_EXPAND_ENTRY_H_
#define XGBOOST_TREE_HIST_EXPAND_ENTRY_H_

#include <algorithm>       // for all_of
#include <ostream>         // for ostream
#include <utility>         // for move
#include <vector>          // for vector

#include "../param.h"      // for SplitEntry, SplitEntryContainer, TrainParam
#include "xgboost/base.h"  // for GradientPairPrecise, bst_node_t

namespace xgboost::tree {
/**
 * \brief Structure for storing tree split candidate.
 */
template <typename Impl>
struct ExpandEntryImpl {
  bst_node_t nid;
  bst_node_t depth;

  [[nodiscard]] float GetLossChange() const {
    return static_cast<Impl const*>(this)->split.loss_chg;
  }
  [[nodiscard]] bst_node_t GetNodeId() const { return nid; }

  static bool ChildIsValid(TrainParam const& param, bst_node_t depth, bst_node_t num_leaves) {
    if (param.max_depth > 0 && depth >= param.max_depth) return false;
    if (param.max_leaves > 0 && num_leaves >= param.max_leaves) return false;
    return true;
  }

  [[nodiscard]] bool IsValid(TrainParam const& param, bst_node_t num_leaves) const {
    return static_cast<Impl const*>(this)->IsValidImpl(param, num_leaves);
  }
};

struct CPUExpandEntry : public ExpandEntryImpl<CPUExpandEntry> {
  SplitEntry split;

  CPUExpandEntry() = default;
  CPUExpandEntry(bst_node_t nidx, bst_node_t depth, SplitEntry split)
      : ExpandEntryImpl{nidx, depth}, split(std::move(split)) {}
  CPUExpandEntry(bst_node_t nidx, bst_node_t depth) : ExpandEntryImpl{nidx, depth} {}

  [[nodiscard]] bool IsValidImpl(TrainParam const& param, bst_node_t num_leaves) const {
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

  friend std::ostream& operator<<(std::ostream& os, CPUExpandEntry const& e) {
    os << "ExpandEntry:\n";
    os << "nidx: " << e.nid << "\n";
    os << "depth: " << e.depth << "\n";
    os << "loss: " << e.split.loss_chg << "\n";
    os << "split:\n" << e.split << std::endl;
    return os;
  }

  /**
   * @brief Copy primitive fields into this, and collect cat_bits into a vector.
   *
   * This is used for allgather.
   *
   * @param that The other entry to copy from
   * @param collected_cat_bits The vector to collect cat_bits
   * @param cat_bits_sizes The sizes of the collected cat_bits
   */
  void CopyAndCollect(CPUExpandEntry const& that, std::vector<uint32_t>* collected_cat_bits,
                      std::vector<std::size_t>* cat_bits_sizes) {
    nid = that.nid;
    depth = that.depth;
    split.CopyAndCollect(that.split, collected_cat_bits, cat_bits_sizes);
  }
};

struct MultiExpandEntry : public ExpandEntryImpl<MultiExpandEntry> {
  SplitEntryContainer<std::vector<GradientPairPrecise>> split;

  MultiExpandEntry() = default;
  MultiExpandEntry(bst_node_t nidx, bst_node_t depth) : ExpandEntryImpl{nidx, depth} {}

  [[nodiscard]] bool IsValidImpl(TrainParam const& param, bst_node_t num_leaves) const {
    if (split.loss_chg <= kRtEps) return false;
    auto is_zero = [](auto const& sum) {
      return std::all_of(sum.cbegin(), sum.cend(),
                         [&](auto const& g) { return g.GetHess() - .0 == .0; });
    };
    if (is_zero(split.left_sum) || is_zero(split.right_sum)) {
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

  friend std::ostream& operator<<(std::ostream& os, MultiExpandEntry const& e) {
    os << "ExpandEntry: \n";
    os << "nidx: " << e.nid << "\n";
    os << "depth: " << e.depth << "\n";
    os << "loss: " << e.split.loss_chg << "\n";
    os << "split cond:" << e.split.split_value << "\n";
    os << "split ind:" << e.split.SplitIndex() << "\n";
    os << "left_sum: [";
    for (auto v : e.split.left_sum) {
      os << v << ", ";
    }
    os << "]\n";

    os << "right_sum: [";
    for (auto v : e.split.right_sum) {
      os << v << ", ";
    }
    os << "]\n";
    return os;
  }

  /**
   * @brief Copy primitive fields into this, and collect cat_bits and gradients into vectors.
   *
   * This is used for allgather.
   *
   * @param that The other entry to copy from
   * @param collected_cat_bits The vector to collect cat_bits
   * @param cat_bits_sizes The sizes of the collected cat_bits
   * @param collected_gradients The vector to collect gradients
   */
  void CopyAndCollect(MultiExpandEntry const& that, std::vector<uint32_t>* collected_cat_bits,
                      std::vector<std::size_t>* cat_bits_sizes,
                      std::vector<GradientPairPrecise>* collected_gradients) {
    nid = that.nid;
    depth = that.depth;
    split.CopyAndCollect(that.split, collected_cat_bits, cat_bits_sizes, collected_gradients);
  }
};
}  // namespace xgboost::tree
#endif  // XGBOOST_TREE_HIST_EXPAND_ENTRY_H_
