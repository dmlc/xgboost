/*!
 * Copyright 2019 XGBoost contributors
 */
#pragma once

#include <thrust/device_vector.h>
#include <vector>
#include <set>

#include "../common/span.h"

namespace xgboost {
namespace tree {

struct InteractionConstraints {
  // this is a mapping to mapping. nid -> feature_id -> interactions belonging to feature_id
  std::vector< std::set<int32_t> > node_interactions_;
  thrust::device_vector<int32_t> d_node_interactions_;
  thrust::device_vector<int32_t> d_node_interactions_ptr_;
  size_t n_node_interactions_ {0};

  std::vector< std::set<int32_t> > feature_interactions_;
  thrust::device_vector<int32_t> d_feature_interactions_;
  thrust::device_vector<int32_t> d_feature_interactions_ptr_;

 public:
  struct DeviceEvaluator {
    common::Span<int32_t> feature_interaction_;
    common::Span<int32_t> feature_interaction_ptr_;
    common::Span<int32_t> node_interactions_span_;
    common::Span<int32_t> node_interactions_ptr_span_;

    // FIXME(trivialfis); gpu_hist uses one block for each feature, how should I integrate it?
    XGBOOST_DEVICE float EvaluateSplit(int32_t nid, int32_t fid, float gain) const {
      if (node_interactions_span_.size() == 0) {
        // interaction constraint is not used.
        return gain;
      }
      if (node_interactions_ptr_span_[nid+1] - node_interactions_ptr_span_[nid] == 0) {
        // no constraint for nid
        return gain;
      }
      for (size_t feature_index = node_interactions_ptr_span_[nid];
           feature_index < node_interactions_ptr_span_[nid+1];
           ++feature_index) {
        // map to corresponding interactions set
        int32_t feature = node_interactions_span_[feature_index];
        common::Span<int32_t> feature_interactions = feature_interaction_.subspan(
            feature_interaction_ptr_[feature],
            feature_interaction_ptr_[feature+1]);
        bool accept = false;
        // search for feature fid.
        // FIXME(trivialfis); Binary search
        for (auto f : feature_interactions) {
          if (fid == f) {
            accept = true;
            break;
          }
        }
        if (accept) {
          // Current feature fid compiles the constraint, return original gain.
          return gain;
        }
      }
      // Current feature fid is not allowed.
      return -std::numeric_limits<bst_float>::infinity();
    }
  } split_evaluator_;

  InteractionConstraints(std::string interaction_constraints_str, int n_features);
  void ApplySplit(int32_t nid, int32_t left, int32_t right, int32_t fid);
};

}  // namespace tree
}  // namespace xgboost