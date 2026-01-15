/**
 * Copyright 2025-2026, XGBoost contributors
 */
#pragma once

#include "../../common/device_vector.cuh"  // for device_vector
#include "evaluate_splits.cuh"             // for MultiEvaluateSplitSharedInputs
#include "xgboost/base.h"                  // for GradientPairInt64
#include "xgboost/context.h"               // for Context

namespace xgboost::tree::cuda_impl {
/** @brief Evaluator for vector leaf. */
class MultiHistEvaluator {
 public:
  struct WeightBuffer {
    // * 3 because of base, left, right weights.
    constexpr static bst_node_t kNodes = 3;

    common::Span<float> weights;
    bst_target_t n_targets;

    static WeightBuffer Make(bst_node_t n_nodes, bst_target_t n_targets,
                             dh::DeviceUVector<float> *p_weights) {
      p_weights->resize(n_nodes * n_targets * kNodes);
      WeightBuffer buf{dh::ToSpan(*p_weights), n_targets};
      return buf;
    }
    // get the base weight buffer
    XGBOOST_DEVICE common::Span<float> Base(std::size_t nidx_in_set) const {
      return weights.subspan(nidx_in_set * n_targets * kNodes, n_targets);
    }
    XGBOOST_DEVICE common::Span<float> Left(std::size_t nidx_in_set) const {
      return weights.subspan(nidx_in_set * n_targets * kNodes + n_targets, n_targets);
    }
    XGBOOST_DEVICE common::Span<float> Right(std::size_t nidx_in_set) const {
      return weights.subspan(nidx_in_set * n_targets * kNodes + n_targets * 2, n_targets);
    }
  };

 private:
  // Buffer for node weights
  dh::DeviceUVector<float> weights_;
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
}  // namespace xgboost::tree::cuda_impl
