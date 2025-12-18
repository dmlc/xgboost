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
/** @brief Evaluator for vector leaf. */
class MultiHistEvaluator {
  // Buffer for node weights
  dh::device_vector<float> weights_;
  // Buffer for histogram scans.
  dh::DeviceUVector<GradientPairInt64> scan_buffer_;
  // Buffer for node gradient sums.
  dh::device_vector<GradientPairInt64> node_sums_;

 public:
  template <typename GradT>
  static XGBOOST_DEVICE common::Span<GradT> GetNodeSumImpl(common::Span<GradT> node_sums,
                                                           bst_node_t nidx,
                                                           bst_target_t n_targets) {
    auto offset = nidx * n_targets;
    return node_sums.subspan(offset, n_targets);
  }
  /**
   * @brief Run evaluation for the root node.
   */
  [[nodiscard]] MultiExpandEntry EvaluateSingleSplit(
      Context const *ctx, MultiEvaluateSplitInputs const &input,
      MultiEvaluateSplitSharedInputs const &shared_inputs);
  /**
   * @brief Run evaluation for multiple nodes.
   */
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
    return GetNodeSumImpl(dh::ToSpan(this->node_sums_), nidx, n_targets);
  }
  [[nodiscard]] common::Span<GradientPairInt64 const> GetNodeSum(bst_node_t nidx,
                                                                 bst_target_t n_targets) const {
    return GetNodeSumImpl(dh::ToSpan(this->node_sums_), nidx, n_targets);
  }

  // Track the child gradient sum.
  void ApplyTreeSplit(Context const *ctx, RegTree const *p_tree,
                      common::Span<MultiExpandEntry const> d_candidates, bst_target_t n_targets);
};

std::ostream &DebugPrintHistogram(std::ostream &os, common::Span<GradientPairInt64 const> node_hist,
                                  common::Span<GradientQuantiser const> roundings,
                                  bst_target_t n_targets);
}  // namespace xgboost::tree::cuda_impl
