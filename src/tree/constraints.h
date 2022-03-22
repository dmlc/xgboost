/*!
 * Copyright 2018-2019 by Contributors
 */
#ifndef XGBOOST_TREE_CONSTRAINTS_H_
#define XGBOOST_TREE_CONSTRAINTS_H_

#include <string>
#include <unordered_set>
#include <vector>

#include "xgboost/span.h"
#include "xgboost/base.h"

#include "param.h"

namespace xgboost {
/*!
 * \brief Feature interaction constraint implementation for CPU tree updaters.
 *
 * The interface is similar to the one for GPU Hist.
 */
class FeatureInteractionConstraintHost {
 protected:
  // interaction_constraints_[constraint_id] contains a single interaction
  //   constraint, which specifies a group of feature IDs that can interact
  //   with each other
  std::vector< std::unordered_set<bst_feature_t> > interaction_constraints_;
  // int_cont_[nid] contains the set of all feature IDs that are allowed to
  //   be used for a split at node nid
  std::vector< std::unordered_set<bst_feature_t> > node_constraints_;
  // splits_[nid] contains the set of all feature IDs that have been used for
  //   splits in node nid and its parents
  std::vector< std::unordered_set<bst_feature_t> > splits_;
  // string passed by user.
  std::string interaction_constraint_str_;
  // number of features in DMatrix/Booster
  bst_feature_t n_features_;
  bool enabled_{false};

  void SplitImpl(int32_t node_id, bst_feature_t feature_id, bst_node_t left_id,
                 bst_node_t right_id);

 public:
  FeatureInteractionConstraintHost() = default;
  void Split(int32_t node_id, bst_feature_t feature_id, bst_node_t left_id,
             bst_node_t right_id) {
    if (!enabled_) {
      return;
    } else {
      this->SplitImpl(node_id, feature_id, left_id, right_id);
    }
  }

  bool Query(bst_node_t nid, bst_feature_t fid) const {
    if (!enabled_) { return true; }
    return node_constraints_.at(nid).find(fid) != node_constraints_.at(nid).cend();
  }

  void Reset();

  void Configure(tree::TrainParam const& param, bst_feature_t const n_features);
};
}  // namespace xgboost

#endif  // XGBOOST_TREE_CONSTRAINTS_H_
