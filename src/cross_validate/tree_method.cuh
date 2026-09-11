/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <algorithm>  // for any_of
#include <memory>     // for unique_ptr
#include <optional>   // for optional
#include <vector>     // for vector

#include "../common/cuda_context.cuh"  // for CUDAContext
#include "../common/device_helpers.cuh"
#include "../tree/driver.h"     // for Driver
#include "../tree/tree_view.h"  // for MultiTargetTreeView
#include "../tree/updater_gpu_hist.cuh"
#include "cross_validate.h"   // for UnitLayout
#include "xgboost/context.h"  // for Context
#include "xgboost/span.h"     // for Span

namespace xgboost::cv {
using tree::cuda_impl::MultiExpandEntry;
using tree::cuda_impl::StaticBatch;
// The partitioning helpers are shared with the single-model GPU hist maker.
using HistMaker = tree::cuda_impl::MultiTargetHistMaker;
template <typename Accessor>
using GoLeftOp = HistMaker::GoLeftOp<Accessor>;
using PartitionNodes = HistMaker::PartitionNodes;

inline void DebugCheckValid(Context const* ctx, bst_idx_t n_expected,
                            common::Span<bst_node_t const> d_position) {
  // Every training row of the unit, and only those, must have received a position.
  auto n_valid = thrust::count_if(
      ctx->CUDACtx()->CTP(), dh::tcbegin(d_position), dh::tcend(d_position),
      [] XGBOOST_DEVICE(bst_node_t nidx) { return nidx != RegTree::kInvalidNodeId; });
  CHECK_EQ(static_cast<bst_idx_t>(n_valid), n_expected);
}

template <template <typename> typename GoLeftOp, typename Acc>
void RouteHeldOut(Context const* ctx, MembershipView membership, FoldId fold,
                  tree::MultiTargetTreeView tree, GoLeftOp<Acc> go_left,
                  common::Span<bst_node_t> in_out_position) {
  dh::LaunchN(membership.ids.size(), ctx->CUDACtx()->Stream(), [=] XGBOOST_DEVICE(std::size_t i) {
    auto ridx = membership.base_rowid + i;
    if (!membership.IsValidation(fold, ridx)) {
      return;
    }
    auto nidx = in_out_position[ridx];
    if (tree.IsLeaf(nidx)) {
      return;
    }
    bool is_left = go_left(ridx, tree::cuda_impl::MultiTargetHistMaker::NodeSplitData{nidx});
    in_out_position[ridx] = is_left ? tree.LeftChild(nidx) : tree.RightChild(nidx);
  });
}

// Everything the tree method keeps for one training unit. The caches at the top survive a
// round so that the next one reuses them; the rest describes the tree being grown now.
struct UnitState {
  std::unique_ptr<tree::DeviceHistogramBuilder> histogram;
  std::unique_ptr<tree::GradientQuantiserGroup> quantizer;
  std::unique_ptr<tree::cuda_impl::MultiHistEvaluator> evaluator;
  linalg::Matrix<GradientPairInt64> quantized_gpair;
  tree::RowPartitionerBatches partitioners;

  std::unique_ptr<RegTree> tree;
  std::unique_ptr<tree::Driver<MultiExpandEntry>> driver;
  // Nodes split at the current level, and the subset of them whose children may split
  // again. The children of the rest are leaves, so their histograms are never built.
  std::vector<MultiExpandEntry> expand_set;
  std::vector<MultiExpandEntry> candidates;
  // Staging for the children the device evaluated, read back once per level.
  dh::PinnedMemory staged;
  dh::DeviceUVector<tree::MultiEvaluateSplitInputs> eval_inputs;
  dh::DeviceUVector<MultiExpandEntry> eval_outputs;
};

// The state of one round: the per-unit state, and the index space that describes it. A unit
// index means the same model here as in every other buffer of the round.
struct FoldTreeState {
  UnitLayout layout;
  bst_target_t n_targets{0};
  std::vector<std::unique_ptr<UnitState>> units;
  // Node position of every held-out row. Not indexed by unit: a row is held out by
  // exactly one fold, so one array serves them all.
  dh::DeviceUVector<bst_node_t> oof_position;

  [[nodiscard]] std::size_t NumUnits() const noexcept(true) { return this->units.size(); }
  [[nodiscard]] UnitState& At(std::size_t u) { return *this->units.at(u); }
  [[nodiscard]] bool IsRefit(std::size_t u) const noexcept(true) { return this->layout.IsRefit(u); }
  // Whether any unit still has a level to grow.
  [[nodiscard]] bool Growing() const {
    return std::any_of(this->units.cbegin(), this->units.cend(),
                       [](auto const& unit) { return !unit->expand_set.empty(); });
  }
  // Ensure there is one state per unit, keeping the caches of the units that already exist,
  // which is the common case across boosting rounds.
  void Resize(std::size_t n_units) {
    while (this->units.size() > n_units) {
      this->units.pop_back();
    }
    while (this->units.size() < n_units) {
      this->units.emplace_back(std::make_unique<UnitState>());
    }
  }
  // Move the finished trees out, one tree per unit, for `FoldModels::CommitModel`.
  [[nodiscard]] std::vector<gbm::TreesOneIter> TakeTrees() {
    std::vector<gbm::TreesOneIter> trees(this->NumUnits());
    for (std::size_t u = 0; u < this->NumUnits(); ++u) {
      auto& unit = this->At(u);
      CHECK(unit.tree);
      trees[u].resize(1);
      trees[u].front().emplace_back(std::move(unit.tree));
    }
    return trees;
  }
};

// The nodes one unit works on at one level. `partition` and `tree_view` are derived from
// the expand set, and an empty `tree_view` is what says the unit takes no part in this
// level.  The subtraction vectors are index-aligned, which is the contract
// `DeviceHistogramBuilder::SubtractHist` expects.
struct LevelNodes {
  PartitionNodes partition{0};
  // `MultiTargetTreeView` has no default constructor, and an inactive unit has no view.
  std::optional<tree::MultiTargetTreeView> tree_view;
  std::vector<bst_node_t> nodes_to_build;
  std::vector<MultiExpandEntry> sub_parent;
  std::vector<bst_node_t> sub_sibling;
  std::vector<bst_node_t> sub;

  [[nodiscard]] bool Active() const noexcept(true) { return this->tree_view.has_value(); }
};
}  // namespace xgboost::cv
