/**
 * Copyright 2025-2026, XGBoost Contributors
 */
#pragma once

#include <xgboost/base.h>  // for bst_target_t

#include <vector>  // for vector

#include "../../../../src/common/device_vector.cuh"     // for device_vector
#include "../../../../src/tree/gpu_hist/quantiser.cuh"  // for GradientQuantiser
#include "../../helpers.h"

namespace xgboost::tree {
inline GradientQuantiser MakeDummyQuantizer() {
  return {GradientPairPrecise{1.0f, 1.0f}, GradientPairPrecise{1.0f, 1.0f}};
}

inline auto MakeDummyQuantizers(bst_target_t n_targets) {
  std::vector<GradientQuantiser> h_quantizers;
  for (bst_target_t i = 0; i < n_targets; ++i) {
    h_quantizers.emplace_back(MakeDummyQuantizer());
  }
  dh::device_vector<GradientQuantiser> d_quantizers(h_quantizers);
  return d_quantizers;
}

inline auto GenerateGradientsFixedPoint(Context const* ctx, size_t n_rows, float lower = 0.0f,
                                        float upper = 1.0f) {
  auto gpairs = GenerateRandomGradients(n_rows, lower, upper);
  gpairs.SetDevice(ctx->Device());
  auto quantiser =
      GradientQuantiser{ctx, linalg::MakeVec(ctx->Device(), gpairs.ConstDeviceSpan()), MetaInfo{}};
  dh::device_vector<GradientQuantiser> roundings{quantiser};
  linalg::Matrix<GradientPairInt64> gpairs_i64;
  CalcQuantizedGpairs(ctx, linalg::MakeTensorView(ctx, gpairs.ConstDeviceSpan(), gpairs.Size(), 1),
                      dh::ToSpan(roundings), &gpairs_i64);
  return gpairs_i64;
}
}  // namespace xgboost::tree
