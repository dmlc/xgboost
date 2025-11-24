/**
 * Copyright 2025, XGBoost Contributors
 */
#pragma once

#include <xgboost/base.h>  // for bst_target_t

#include <vector>  // for vector

#include "../../../../src/common/device_vector.cuh"     // for device_vector
#include "../../../../src/tree/gpu_hist/quantiser.cuh"  // for GradientQuantiser

namespace xgboost::tree {
inline auto MakeDummyQuantizers(bst_target_t n_targets) {
  std::vector<GradientQuantiser> h_quantizers;
  for (bst_target_t i = 0; i < n_targets; ++i) {
    h_quantizers.emplace_back(GradientPairPrecise{1.0f, 1.0f}, GradientPairPrecise{1.0f, 1.0f});
  }
  dh::device_vector<GradientQuantiser> d_quantizers(h_quantizers);
  return d_quantizers;
}
}  // namespace xgboost::tree
