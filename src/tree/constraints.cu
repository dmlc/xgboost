/*!
 * Copyright 2019 XGBoost contributors
 */
#include <thrust/copy.h>
#include <memory>
#include "constraints.cuh"
#include "../common/device_helpers.cuh"
#include "../common/span.h"

namespace xgboost {
namespace tree {

InteractionConstraints::InteractionConstraints(std::string interaction_constraints_str, int n_features) {
  if (interaction_constraints_str.size() == 0) {
    is_used = false;
    return;
  }
  std::istringstream iss(interaction_constraints_str);
  dmlc::JSONReader reader(&iss);
  // Read std::vector<std::vector<bst_uint>> first and then
  //   convert to std::vector<std::unordered_set<bst_uint>>
  std::vector<std::vector<bst_uint>> tmp;
  reader.Read(&tmp);

  // Initialize feature -> interaction mapping
  feature_interactions_.resize(n_features);
  for (const auto& constraints : tmp) {
    for (auto feature : constraints) {
      for (auto e : constraints) {
        feature_interactions_[feature].insert(e);
      }
    }
  }

  size_t total_size = 0;
  for (const auto& constraints : feature_interactions_) {
    total_size += constraints.size();
  }

  std::vector<int32_t> h_feature_interactions (total_size);
  std::vector<int32_t> h_feature_interactions_ptr;
  size_t pointer = 0;
  h_feature_interactions_ptr.emplace_back(pointer);
  for (size_t i = 0; i < feature_interactions_.size(); ++i) {
    std::copy(feature_interactions_[i].cbegin(), feature_interactions_[i].cend(),
              h_feature_interactions.begin() + pointer);
    pointer += feature_interactions_[i].size();
    h_feature_interactions_ptr.emplace_back(pointer);
  }
  d_feature_interactions_ptr_.resize(h_feature_interactions_ptr.size());
  thrust::copy(h_feature_interactions_ptr.cbegin(), h_feature_interactions_ptr.cend(),
               d_feature_interactions_ptr_.begin());
  d_feature_interactions_.resize(h_feature_interactions.size());
  thrust::copy(h_feature_interactions.cbegin(), h_feature_interactions.cend(),
               d_feature_interactions_.begin());

  split_evaluator_.feature_interaction_ = dh::ToSpan(d_feature_interactions_);
  split_evaluator_.feature_interaction_ptr_ = dh::ToSpan(d_feature_interactions_ptr_);
}

void InteractionConstraints::ApplySplit(int32_t nid, int32_t left, int32_t right, int32_t fid) {
  int32_t max_size = std::max(left, right);
  max_size = std::max(nid, max_size) + 1;
  node_interactions_.resize(max_size);

  node_interactions_[left] = node_interactions_[nid];
  node_interactions_[right] = node_interactions_[nid];
  // Permit previous feature.
  node_interactions_[left].insert(fid);
  node_interactions_[right].insert(fid);

  size_t n_features_in_node = 0;
  for (auto const& interactions : node_interactions_) {
    n_features_in_node += interactions.size();
  }

  std::vector<int32_t> h_node_interactions (n_features_in_node);
  std::vector<int32_t> h_node_interactions_ptr {0};

  int32_t count = 0;
  for (size_t i = 0; i < node_interactions_.size(); ++i) {
    CHECK_LT(count, h_node_interactions.size());
    std::copy(node_interactions_[i].begin(), node_interactions_[i].end(),
              h_node_interactions.begin() + count);
    count += node_interactions_[i].size();
    h_node_interactions_ptr.push_back(count);
  }

  d_node_interactions_.resize(h_node_interactions.size());
  thrust::copy(h_node_interactions.cbegin(), h_node_interactions.cend(),
               d_node_interactions_.begin());
  d_node_interactions_ptr_.resize(h_node_interactions_ptr.size());
  thrust::copy(h_node_interactions_ptr.cbegin(), h_node_interactions_ptr.cend(),
               d_node_interactions_ptr_.begin());

  split_evaluator_.node_interactions_span_ = dh::ToSpan(d_node_interactions_);
  split_evaluator_.node_interactions_ptr_span_ = dh::ToSpan(d_node_interactions_ptr_);

  feature_buffer = std::make_shared<HostDeviceVector<int>>();
}
}  // namespace tree
}  // namespace xgboost