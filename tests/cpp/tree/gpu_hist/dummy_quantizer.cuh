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

struct QuantizedGradients {
  linalg::Matrix<GradientPairInt64> gpair;
  MultiGradientQuantiser quantizer;
};

// Returns both quantized gradients and quantizers.
inline auto GenerateGradientsFixedPoint(Context const* ctx, bst_idx_t n_samples,
                                        bst_target_t n_targets = 1, float lower = 0.0f,
                                        float upper = 1.0f) {
  auto gpairs = GenerateRandomGradients(n_samples * n_targets, lower, upper);
  gpairs.SetDevice(ctx->Device());
  auto d_gpair = linalg::MakeTensorView(ctx, gpairs.ConstDeviceSpan(), n_samples, n_targets);

  // Create a quantizer per target
  MultiGradientQuantiser multi_quantizer{ctx, d_gpair, MetaInfo{}};

  linalg::Matrix<GradientPairInt64> gpairs_i64;
  CalcQuantizedGpairs(ctx, d_gpair, multi_quantizer.Quantizers(), &gpairs_i64);

  return QuantizedGradients{std::move(gpairs_i64), std::move(multi_quantizer)};
}
}  // namespace xgboost::tree
