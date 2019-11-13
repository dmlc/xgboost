/*!
 * Copyright 2018-2019 by Contributors
 */
#include <algorithm>
#include <unordered_set>
#include <vector>

#include "xgboost/span.h"
#include "constraints.h"
#include "param.h"

namespace xgboost {
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
  // Parse interaction constraints
  std::istringstream iss(this->interaction_constraint_str_);
  dmlc::JSONReader reader(&iss);
  // Read std::vector<std::vector<bst_uint>> first and then
  //   convert to std::vector<std::unordered_set<bst_uint>>
  std::vector<std::vector<bst_uint>> tmp;
  try {
    reader.Read(&tmp);
  } catch (dmlc::Error const& e) {
    LOG(FATAL) << "Failed to parse feature interaction constraint:\n"
               << this->interaction_constraint_str_ << "\n"
               << "With error:\n" << e.what();
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

void FeatureInteractionConstraintHost::SplitImpl(
    bst_node_t node_id, bst_feature_t feature_id, bst_node_t left_id, bst_node_t right_id) {
  bst_node_t newsize = std::max(left_id, right_id) + 1;

  // Record previous splits for child nodes
  auto feature_splits = splits_[node_id];  // fid history of current node
  feature_splits.insert(feature_id);  // add feature of current node
  splits_.resize(newsize);
  splits_[left_id] = feature_splits;
  splits_[right_id] = feature_splits;

  // Resize constraints record, initialise all features to be not permitted for new nodes
  CHECK_NE(newsize, 0);
  node_constraints_.resize(newsize, std::unordered_set<bst_feature_t>());

  // Permit features used in previous splits
  for (bst_feature_t fid : feature_splits) {
    node_constraints_[left_id].insert(fid);
    node_constraints_[right_id].insert(fid);
  }

  // Loop across specified interactions in constraints
  for (const auto &constraint : interaction_constraints_) {
    // flags whether the specified interaction is still relevant
    bst_uint flag = 1;

    // Test relevance of specified interaction by checking all previous
    // features are included
    for (bst_uint checkvar : feature_splits) {
      if (constraint.count(checkvar) == 0) {
        flag = 0;
        break;   // interaction is not relevant due to unmet constraint
      }
    }

    // If interaction is still relevant, permit all other features in the
    // interaction
    if (flag == 1) {
      for (bst_uint k : constraint) {
        node_constraints_[left_id].insert(k);
        node_constraints_[right_id].insert(k);
      }
    }
  }
}
}  // namespace xgboost
