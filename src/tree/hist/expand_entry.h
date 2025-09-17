/**
 * Copyright 2021-2023, XGBoost Contributors
 */
#ifndef XGBOOST_TREE_HIST_EXPAND_ENTRY_H_
#define XGBOOST_TREE_HIST_EXPAND_ENTRY_H_

#include <algorithm>    // for all_of
#include <ostream>      // for ostream
#include <string>       // for string
#include <type_traits>  // for add_const_t
#include <utility>      // for move
#include <vector>       // for vector

#include "../../common/type.h"  // for EraseType
#include "../param.h"           // for SplitEntry, SplitEntryContainer, TrainParam
#include "xgboost/base.h"       // for GradientPairPrecise, bst_node_t
#include "xgboost/json.h"       // for Json

namespace xgboost::tree {
/**
 * \brief Structure for storing tree split candidate.
 */
template <typename Impl>
struct ExpandEntryImpl {
  bst_node_t nid{0};
  bst_node_t depth{0};

  [[nodiscard]] float GetLossChange() const {
    return static_cast<Impl const*>(this)->split.loss_chg;
  }
  [[nodiscard]] bst_node_t GetNodeId() const { return nid; }

  [[nodiscard]] bool IsValid(TrainParam const& param, bst_node_t num_leaves) const {
    return static_cast<Impl const*>(this)->IsValidImpl(param, num_leaves);
  }

  void Save(Json* p_out) const {
    auto& out = *p_out;
    auto self = static_cast<Impl const*>(this);

    out["nid"] = Integer{this->nid};
    out["depth"] = Integer{this->depth};

    /**
     * Handle split
     */
    out["split"] = Object{};
    auto& split = out["split"];
    split["loss_chg"] = self->split.loss_chg;
    split["sindex"] = Integer{self->split.sindex};
    split["split_value"] = self->split.split_value;

    auto const& cat_bits = self->split.cat_bits;
    auto s_cat_bits = common::Span{cat_bits.data(), cat_bits.size()};
    split["cat_bits"] = U8Array{s_cat_bits.size_bytes()};
    auto& j_cat_bits = get<U8Array>(split["cat_bits"]);
    using T = typename decltype(self->split.cat_bits)::value_type;
    auto erased =
        common::EraseType<std::add_const_t<T>, std::add_const_t<std::uint8_t>>(s_cat_bits);
    for (std::size_t i = 0; i < erased.size(); ++i) {
      j_cat_bits[i] = erased[i];
    }

    split["is_cat"] = Boolean{self->split.is_cat};

    self->SaveGrad(&split);
  }

  void Load(Json const& in) {
    auto self = static_cast<Impl*>(this);

    this->nid = get<Integer const>(in["nid"]);
    this->depth = get<Integer const>(in["depth"]);

    /**
     * Handle split
     */
    auto const& split = in["split"];
    self->split.loss_chg = get<Number const>(split["loss_chg"]);
    self->split.sindex = get<Integer const>(split["sindex"]);
    self->split.split_value = get<Number const>(split["split_value"]);

    auto const& j_cat_bits = get<U8Array const>(split["cat_bits"]);
    using T = typename decltype(self->split.cat_bits)::value_type;
    auto restored = common::RestoreType<std::add_const_t<T>>(
        common::Span{j_cat_bits.data(), j_cat_bits.size()});
    self->split.cat_bits.resize(restored.size());
    for (std::size_t i = 0; i < restored.size(); ++i) {
      self->split.cat_bits[i] = restored[i];
    }

    self->split.is_cat = get<Boolean const>(split["is_cat"]);
    self->LoadGrad(split);
  }
};

struct CPUExpandEntry : public ExpandEntryImpl<CPUExpandEntry> {
  SplitEntry split;

  CPUExpandEntry() = default;
  CPUExpandEntry(bst_node_t nidx, bst_node_t depth, SplitEntry split)
      : ExpandEntryImpl{nidx, depth}, split(std::move(split)) {}
  CPUExpandEntry(bst_node_t nidx, bst_node_t depth) : ExpandEntryImpl{nidx, depth} {}

  void SaveGrad(Json* p_out) const {
    auto& out = *p_out;
    auto save = [&](std::string const& name, GradStats const& sum) {
      out[name] = F64Array{2};
      auto& array = get<F64Array>(out[name]);
      array[0] = sum.GetGrad();
      array[1] = sum.GetHess();
    };
    save("left_sum", this->split.left_sum);
    save("right_sum", this->split.right_sum);
  }
  void LoadGrad(Json const& in) {
    auto const& left_sum = get<F64Array const>(in["left_sum"]);
    this->split.left_sum = GradStats{left_sum[0], left_sum[1]};
    auto const& right_sum = get<F64Array const>(in["right_sum"]);
    this->split.right_sum = GradStats{right_sum[0], right_sum[1]};
  }

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

  void SaveGrad(Json* p_out) const {
    auto& out = *p_out;
    auto save = [&](std::string const& name, std::vector<GradientPairPrecise> const& sum) {
      out[name] = F64Array{sum.size() * 2};
      auto& array = get<F64Array>(out[name]);
      for (std::size_t i = 0, j = 0; i < sum.size(); i++, j += 2) {
        array[j] = sum[i].GetGrad();
        array[j + 1] = sum[i].GetHess();
      }
    };
    save("left_sum", this->split.left_sum);
    save("right_sum", this->split.right_sum);
  }
  void LoadGrad(Json const& in) {
    auto load = [&](std::string const& name, std::vector<GradientPairPrecise>* p_sum) {
      auto const& array = get<F64Array const>(in[name]);
      auto& sum = *p_sum;
      sum.resize(array.size() / 2);
      for (std::size_t i = 0, j = 0; i < sum.size(); ++i, j += 2) {
        sum[i] = GradientPairPrecise{array[j], array[j + 1]};
      }
    };
    load("left_sum", &this->split.left_sum);
    load("right_sum", &this->split.right_sum);
  }

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
