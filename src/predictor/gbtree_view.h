/**
 * Copyright 2025, XGBoost Contributors
 */
#pragma once

#include <mutex>    // for mutex, lock_guard
#include <utility>  // for move
#include <vector>   // for vector

#include "../gbm/gbtree_model.h"  // for GBTreeModel
#include "../tree/tree_view.h"    // for MultiTargetTreeView, ScalarTreeView
#include "xgboost/base.h"         // for bst_tree_t, bst_target_t
#include "xgboost/context.h"      // for Context
#include "xgboost/logging.h"      // for CHECK_GT
#include "xgboost/span.h"         // for Span

namespace xgboost::predictor {
// A view for the boosted trees to ensure thread safety.
template <template <typename> typename Container, typename TreeViewVar, typename CopyViews>
struct GBTreeModelView {
 private:
  Container<TreeViewVar> trees_;

 public:
  bst_tree_t tree_begin;
  bst_tree_t tree_end;
  common::Span<bst_target_t const> tree_groups;
  bst_target_t n_groups;
  bst_feature_t n_features;
  bst_node_t n_nodes{0};

 public:
  explicit GBTreeModelView(Context const* ctx, gbm::GBTreeModel const& model, bst_tree_t tree_begin,
                           bst_tree_t tree_end, std::mutex* p_mu)
      : tree_begin{tree_begin},
        tree_end{tree_end},
        n_groups{model.learner_model_param->OutputLength()},
        n_features{model.learner_model_param->num_feature} {
    // Make sure the trees are pulled to target device without race.
    std::lock_guard guard{*p_mu};
    // Copy tree views.
    std::vector<TreeViewVar> trees;
    for (bst_tree_t tree_idx = this->tree_begin; tree_idx < this->tree_end; ++tree_idx) {
      auto const& p_tree = model.trees[tree_idx];
      if (p_tree->IsMultiTarget()) {
        auto d_tree = tree::MultiTargetTreeView{ctx, p_tree.get()};
        this->n_nodes += d_tree.Size();
        trees.emplace_back(d_tree);
      } else {
        auto d_tree = tree::ScalarTreeView{ctx, p_tree.get()};
        this->n_nodes += d_tree.Size();
        trees.emplace_back(d_tree);
      }
    }

    CopyViews::Copy(ctx, &this->trees_, std::move(trees));

    CHECK_GT(this->tree_end, this->tree_begin);
    auto n_trees = this->tree_end - this->tree_begin;
    model.tree_info.SetDevice(ctx->Device());
    this->tree_groups = model.TreeGroups(ctx->Device()).subspan(this->tree_begin, n_trees);
    CHECK_EQ(n_trees, this->trees_.size());
  }

  common::Span<TreeViewVar const> Trees() const { return {trees_.data(), trees_.size()}; }

  GBTreeModelView() = delete;
  GBTreeModelView(GBTreeModelView const&) = delete;
  GBTreeModelView& operator=(GBTreeModelView const&) = delete;
  GBTreeModelView(GBTreeModelView&&) = default;
  GBTreeModelView& operator=(GBTreeModelView&&) = delete;
};
}  // namespace xgboost::predictor
