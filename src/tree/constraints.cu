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
}

std::shared_ptr<HostDeviceVector<int32_t>> InteractionConstraints::GetAllowedFeatures(
    std::shared_ptr<HostDeviceVector<int32_t>> feature_set, int32_t nid) {
  if (!is_used) {
    // interaction constraint is not used.
    return feature_set;
  }
  auto& node_features_constraints = node_interactions_.at(nid);
  if (node_features_constraints.size() == 0) {
    return feature_set;
  }
  auto& h_feature_set = feature_set->HostVector();
  auto& h_feature_buffer = feature_buffer->HostVector();
  h_feature_buffer.clear();

  // evaluate each feature from input
  for (auto fid : h_feature_set) {
    for (auto features_index : node_features_constraints) {
      auto const& features = feature_interactions_.at(features_index);
      if (features.find(fid) != features.cend()) {
        h_feature_buffer.push_back(fid);
      }
    }
  }
  return feature_buffer;
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

  feature_buffer = std::make_shared<HostDeviceVector<int>>();
}
}  // namespace tree
}  // namespace xgboost