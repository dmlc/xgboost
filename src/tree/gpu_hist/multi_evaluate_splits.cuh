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
  dh::device_vector<GradientPairInt64> node_sums_;

 public:
  [[nodiscard]] MultiExpandEntry EvaluateSingleSplit(
      Context const *ctx, MultiEvaluateSplitInputs const &input,
      MultiEvaluateSplitSharedInputs const &shared_inputs);

  void EvaluateSplits(Context const *ctx, common::Span<MultiEvaluateSplitInputs const> d_inputs,
                      MultiEvaluateSplitSharedInputs const &shared_inputs,
                      common::Span<MultiExpandEntry> out_splits);

  void AllocNodeSum(bst_node_t nidx, bst_target_t n_targets) {
    auto end = (nidx + 1) * n_targets;
    if (this->node_sums_.size() < end) {
      this->node_sums_.resize(end);
    }
  }
  [[nodiscard]] common::Span<GradientPairInt64> GetNodeSum(bst_node_t nidx,
                                                           bst_target_t n_targets) {
    auto offset = nidx * n_targets;
    return dh::ToSpan(this->node_sums_).subspan(offset, n_targets);
  }
  [[nodiscard]] common::Span<GradientPairInt64 const> GetNodeSum(bst_node_t nidx,
                                                                 bst_target_t n_targets) const {
    auto offset = nidx * n_targets;
    return dh::ToSpan(this->node_sums_).subspan(offset, n_targets);
  }

  // Track the child gradient sum.
  void ApplyTreeSplit(Context const *ctx, RegTree const *p_tree, MultiExpandEntry const &candidate);
};

void DebugPrintHistogram(common::Span<GradientPairInt64 const> node_hist,
                         common::Span<GradientQuantiser const> roundings, bst_target_t n_targets);
}  // namespace xgboost::tree::cuda_impl
