/**
 * Copyright 2023 by XGBoost Contributors
 */
#include "histogram.h"

#include <cstddef>  // for size_t
#include <numeric>  // for accumulate
#include <utility>  // for swap
#include <vector>   // for vector

#include "../../common/transform_iterator.h"  // for MakeIndexTransformIter
#include "expand_entry.h"                     // for MultiExpandEntry, CPUExpandEntry
#include "xgboost/logging.h"                  // for CHECK_NE
#include "xgboost/span.h"                     // for Span
#include "xgboost/tree_model.h"               // for RegTree

namespace xgboost::tree {
void AssignNodes(RegTree const *p_tree, std::vector<MultiExpandEntry> const &valid_candidates,
                 common::Span<bst_node_t> nodes_to_build, common::Span<bst_node_t> nodes_to_sub) {
  CHECK_EQ(nodes_to_build.size(), valid_candidates.size());

  std::size_t n_idx = 0;
  for (auto const &c : valid_candidates) {
    auto left_nidx = p_tree->LeftChild(c.nid);
    auto right_nidx = p_tree->RightChild(c.nid);

    auto build_nidx = left_nidx;
    auto subtract_nidx = right_nidx;
    auto lit =
        common::MakeIndexTransformIter([&](auto i) { return c.split.left_sum[i].GetHess(); });
    auto left_sum = std::accumulate(lit, lit + c.split.left_sum.size(), .0);
    auto rit =
        common::MakeIndexTransformIter([&](auto i) { return c.split.right_sum[i].GetHess(); });
    auto right_sum = std::accumulate(rit, rit + c.split.right_sum.size(), .0);
    auto fewer_right = right_sum < left_sum;
    if (fewer_right) {
      std::swap(build_nidx, subtract_nidx);
    }
    nodes_to_build[n_idx] = build_nidx;
    nodes_to_sub[n_idx] = subtract_nidx;
    ++n_idx;
  }
}

void AssignNodes(RegTree const *p_tree, std::vector<CPUExpandEntry> const &candidates,
                 common::Span<bst_node_t> nodes_to_build, common::Span<bst_node_t> nodes_to_sub) {
  std::size_t n_idx = 0;
  for (auto const &c : candidates) {
    auto left_nidx = (*p_tree)[c.nid].LeftChild();
    auto right_nidx = (*p_tree)[c.nid].RightChild();
    auto fewer_right = c.split.right_sum.GetHess() < c.split.left_sum.GetHess();

    auto build_nidx = left_nidx;
    auto subtract_nidx = right_nidx;
    if (fewer_right) {
      std::swap(build_nidx, subtract_nidx);
    }
    nodes_to_build[n_idx] = build_nidx;
    nodes_to_sub[n_idx] = subtract_nidx;
    ++n_idx;
  }
}
}  // namespace xgboost::tree
