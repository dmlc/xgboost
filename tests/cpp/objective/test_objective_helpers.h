/**
 * Copyright 2026, XGBoost contributors
 */
#pragma once

#include <xgboost/base.h>                // for bst_node_t, bst_idx_t
#include <xgboost/data.h>                // for MetaInfo
#include <xgboost/host_device_vector.h>  // for HostDeviceVector

#include <numeric>  // for iota
#include <vector>   // for vector

namespace xgboost {
// Generate node position for two nodes.
inline auto MakePositionsForTest(bst_idx_t n_samples, bst_node_t left_nidx, bst_node_t right_nidx,
                                 HostDeviceVector<bst_node_t>* p_position) {
  HostDeviceVector<bst_node_t>& position = *p_position;
  position.Resize(n_samples, 0);
  auto& h_position = position.HostVector();
  for (size_t i = 0; i < n_samples; ++i) {
    if (i < n_samples / 2) {
      h_position[i] = left_nidx;
    } else {
      h_position[i] = right_nidx;
    }
  }
}

inline void MakeIotaLabelsForTest(bst_idx_t n_samples, bst_target_t n_targets, MetaInfo* p_info) {
  auto& info = *p_info;
  std::vector<float> labels(n_samples * n_targets);
  std::iota(labels.begin(), labels.end(), 0.0f);
  info.labels.Reshape(n_samples, n_targets);
  info.labels.Data()->HostVector() = labels;
  info.num_row_ = n_samples;
}
}  // namespace xgboost
