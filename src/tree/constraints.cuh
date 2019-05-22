/*!
 * Copyright 2019 XGBoost contributors
 */
#pragma once

#include <thrust/device_vector.h>
#include <vector>
#include <set>

#include "../common/span.h"
#include "../common/host_device_vector.h"

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

  std::shared_ptr<HostDeviceVector<int32_t>> feature_buffer;

  bool is_used {true};

 public:
  struct DeviceEvaluator {
    common::Span<int32_t> feature_interaction_;
    common::Span<int32_t> feature_interaction_ptr_;
    common::Span<int32_t> node_interactions_span_;
    common::Span<int32_t> node_interactions_ptr_span_;

    // FIXME(trivialfis); gpu_hist uses one block for each feature, how should I integrate it?
    __device__ float EvaluateSplit(int32_t nid, int32_t fid, float gain) const {
      if (node_interactions_span_.size() == 0) {
        // interaction constraint is not used.
        return gain;
      }
      if (node_interactions_ptr_span_[nid+1] - node_interactions_ptr_span_[nid] == 0) {
        // no constraint for nid
        return gain;
      }
      auto node_interactions = node_interactions_ptr_span_.subspan(nid, 1);

      // FIXME: Unroll this into threads in a block?
      for (size_t feature_index : node_interactions) {
        // map to corresponding interactions set
        int32_t feature = node_interactions_span_[feature_index];
        common::Span<int32_t> feature_interactions = feature_interaction_.subspan(
            feature_interaction_ptr_[feature],
            feature_interaction_ptr_[feature+1] - feature_interaction_ptr_[feature]);
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

  std::shared_ptr<HostDeviceVector<int32_t>> GetAllowedFeatures(
      std::shared_ptr<HostDeviceVector<int32_t>> feature_set, int32_t nid) {
    if (is_used == 0) {
      // interaction constraint is not used.
      return feature_set;
    }
    auto& h_feature_set = feature_set->HostVector();
    auto& h_feature_buffer = feature_buffer->HostVector();
    h_feature_buffer.clear();
    for (auto feat : h_feature_set) {
      if (node_interactions_.at(nid).find(feat) != node_interactions_.at(nid).cend()) {
        h_feature_buffer.push_back(feat);
      }
    }

    return feature_buffer;
  }

  InteractionConstraints(std::string interaction_constraints_str, int n_features);
  void ApplySplit(int32_t nid, int32_t left, int32_t right, int32_t fid);
};

}  // namespace tree
}  // namespace xgboost