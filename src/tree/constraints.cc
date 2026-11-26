/**
 * Copyright 2018-2026, XGBoost Contributors
 */
#include "constraints.h"

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "param.h"

namespace xgboost {
namespace detail {
void CheckFeatureInteractionSplit(bst_node_t nidx, bst_node_t left_idx, bst_node_t right_idx) {
  CHECK_NE(nidx, left_idx) << " Split node: " << nidx << " and its left child: " << left_idx
                           << " cannot be the same.";
  CHECK_NE(nidx, right_idx) << " Split node: " << nidx << " and its right child: " << right_idx
                            << " cannot be the same.";
}
}  // namespace detail

void FeatureInteractionConstraintHost::Configure(tree::TrainParam const& param,
                                                 bst_feature_t const n_features) {
  if (param.interaction_constraints.empty()) {
    enabled_ = !param.interaction_constraints.empty();
    return;  // short-circuit if no constraint is specified
  }
  enabled_ = true;

  this->interaction_constraint_str_ = param.interaction_constraints;
  this->n_features_ = n_features;
  this->Reset();
}

void FeatureInteractionConstraintHost::Reset() {
  if (!enabled_) {
    return;
  }
  // Read std::vector<std::vector<bst_feature_t>> first and then
  //   convert to std::vector<std::unordered_set<bst_feature_t>>
  std::vector<std::vector<bst_feature_t>> tmp;
  try {
    ParseInteractionConstraint(this->interaction_constraint_str_, &tmp);
  } catch (dmlc::Error const& e) {
    LOG(FATAL) << "Failed to parse feature interaction constraint:\n"
               << this->interaction_constraint_str_ << "\n"
               << "With error:\n"
               << e.what();
  }
  for (const auto& e : tmp) {
    interaction_constraints_.emplace_back(e.begin(), e.end());
  }

  // Initialise interaction constraints record with all variables permitted for the first node
  node_constraints_.clear();
  node_constraints_.resize(1, std::unordered_set<bst_feature_t>());
  node_constraints_[0].reserve(n_features_);
  for (bst_feature_t i = 0; i < n_features_; ++i) {
    node_constraints_[0].insert(i);
  }

  // Initialise splits record
  splits_.clear();
  splits_.resize(1, std::unordered_set<bst_feature_t>());
}

void FeatureInteractionConstraintHost::SplitImpl(bst_node_t nidx, bst_feature_t fidx,
                                                 bst_node_t left_idx, bst_node_t right_idx) {
  detail::CheckFeatureInteractionSplit(nidx, left_idx, right_idx);

  bst_node_t newsize = std::max(left_idx, right_idx) + 1;

  // Record previous splits for child nodes
  auto feature_splits = splits_[nidx];  // fidx history of current node
  feature_splits.insert(fidx);          // add feature of current node
  splits_.resize(newsize);
  splits_[left_idx] = feature_splits;
  splits_[right_idx] = feature_splits;

  // Resize constraints record, initialise all features to be not permitted for new nodes
  CHECK_GT(newsize, 0);
  node_constraints_.resize(newsize, std::unordered_set<bst_feature_t>{});

  // Permit features used in previous splits
  for (bst_feature_t fid : feature_splits) {
    node_constraints_[left_idx].insert(fid);
    node_constraints_[right_idx].insert(fid);
  }

  // Loop across specified interactions in constraints
  for (auto const& interaction_group : interaction_constraints_) {
    // Whether the specified interaction is still relevant
    bool flag = true;

    // Test relevance of specified interaction by checking all previous features are
    // included.
    for (bst_feature_t check_fidx : feature_splits) {
      if (interaction_group.count(check_fidx) == 0) {
        flag = false;
        break;  // interaction is not relevant due to unmet constraint
      }
    }

    // If the interaction is still relevant, permit all other features in it.
    if (flag) {
      for (auto k : interaction_group) {
        node_constraints_[left_idx].insert(k);
        node_constraints_[right_idx].insert(k);
      }
    }
  }
}
}  // namespace xgboost
