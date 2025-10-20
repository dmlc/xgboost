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
#include "xgboost/context.h"      // for DeviceOrd
#include "xgboost/logging.h"      // for CHECK_GT
#include "xgboost/span.h"         // for Span

namespace xgboost::predictor {
/**
 * @brief A view for the boosted trees to ensure thread safety.
 *
 *   This class contains a subset of trees based on the input tree range.
 *
 * @tparam Container   The container for storing the tree view variants.
 * @tparam TreeViewVar A std::variant for different view types.
 * @tparam CopyViews   A policy for how to copy the tree views into the container.
 */
template <template <typename> typename Container, typename TreeViewVar, typename CopyViews>
class GBTreeModelView {
 private:
  Container<TreeViewVar> trees_;

 public:
  bst_tree_t const tree_begin;
  bst_tree_t const tree_end;
  common::Span<bst_target_t const> tree_groups;
  bst_target_t const n_groups;
  bst_feature_t const n_features;
  bst_node_t n_nodes{0};

 public:
  explicit GBTreeModelView(DeviceOrd device, gbm::GBTreeModel const& model, bst_tree_t tree_begin,
                           bst_tree_t tree_end, std::mutex* p_mu, CopyViews&& copy)
      : tree_begin{tree_begin},
        tree_end{tree_end},
        n_groups{model.learner_model_param->OutputLength()},
        n_features{model.learner_model_param->num_feature} {
    // Make sure the trees are pulled to target device without race.
    std::lock_guard guard{*p_mu};
    // Create tree views.
    std::vector<TreeViewVar> trees;
    for (bst_tree_t tree_idx = this->tree_begin; tree_idx < this->tree_end; ++tree_idx) {
      auto const& p_tree = model.trees[tree_idx];
      if (p_tree->IsMultiTarget()) {
        auto tree = tree::MultiTargetTreeView{device, p_tree.get()};
        this->n_nodes += tree.Size();
        trees.emplace_back(tree);
      } else {
        auto tree = tree::ScalarTreeView{device, p_tree.get()};
        this->n_nodes += tree.Size();
        trees.emplace_back(tree);
      }
    }

    copy(&this->trees_, std::move(trees));  // NOLINT[build/include_what_you_use]

    CHECK_GE(this->tree_end, this->tree_begin);
    auto n_trees = this->tree_end - this->tree_begin;
    model.tree_info.SetDevice(device);
    this->tree_groups = model.TreeGroups(device).subspan(this->tree_begin, n_trees);
    CHECK_EQ(n_trees, this->trees_.size());
  }

  [[nodiscard]] common::Span<TreeViewVar const> Trees() const {
    return {trees_.data(), trees_.size()};
  }

  GBTreeModelView() = delete;
  GBTreeModelView(GBTreeModelView const&) = delete;
  GBTreeModelView& operator=(GBTreeModelView const&) = delete;
  GBTreeModelView(GBTreeModelView&&) = default;
  GBTreeModelView& operator=(GBTreeModelView&&) = delete;
};
}  // namespace xgboost::predictor
