/**
 * Copyright 2025, XGBoost contributors
 */
#pragma once

#include "../../common/device_vector.cuh"  // for device_vector
#include "evaluate_splits.cuh"             // for MultiEvaluateSplitSharedInputs
#include "quantiser.cuh"                   // for GradientQuantiser
#include "xgboost/base.h"                  // for GradientPairInt64
#include "xgboost/context.h"               // for Context

namespace xgboost::tree::cuda_impl {
class MultiHistEvaluator {
  dh::device_vector<float> weights_;

  dh::device_vector<GradientPairInt64> scan_buffer_;

 public:
  [[nodiscard]] MultiExpandEntry EvaluateSingleSplit(Context const *ctx,
                                                     MultiEvaluateSplitInputs input,
                                                     MultiEvaluateSplitSharedInputs shared_inputs);
};

void DebugPrintHistogram(common::Span<GradientPairInt64 const> node_hist,
                         common::Span<GradientQuantiser const> roundings, bst_target_t n_targets);
}  // namespace xgboost::tree::cuda_impl
