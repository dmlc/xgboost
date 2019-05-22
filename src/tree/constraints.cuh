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
  // this is a mapping to mapping. (nid) -> (feature_id) -> (interactions belonging to feature_id)
  std::vector< std::set<int32_t> > node_interactions_;
  std::vector< std::set<int32_t> > feature_interactions_;
  std::shared_ptr<HostDeviceVector<int32_t>> feature_buffer;

  bool is_used {true};

 public:
  InteractionConstraints(std::string interaction_constraints_str, int n_features);

  std::shared_ptr<HostDeviceVector<int32_t>> GetAllowedFeatures(
      std::shared_ptr<HostDeviceVector<int32_t>> feature_set, int32_t nid);
  void ApplySplit(int32_t nid, int32_t left, int32_t right, int32_t fid);
};

}  // namespace tree
}  // namespace xgboost