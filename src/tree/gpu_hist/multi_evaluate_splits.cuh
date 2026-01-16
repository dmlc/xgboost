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
  /** @brief Buffer to access node weights indexed by node id. */
  struct NodeWeightBuffer {
    // * 3 because of base, left, right weights per node.
    constexpr static bst_node_t kWeightsPerNode = 3;

    common::Span<float> weights;
    bst_target_t n_targets;

    // Get the base weight buffer for a node
    [[nodiscard]] XGBOOST_DEVICE common::Span<float> Base(bst_node_t nidx) const {
      return weights.subspan(nidx * n_targets * kWeightsPerNode, n_targets);
    }
    // Get the left child weight buffer for a node
    [[nodiscard]] XGBOOST_DEVICE common::Span<float> Left(bst_node_t nidx) const {
      return weights.subspan(nidx * n_targets * kWeightsPerNode + n_targets, n_targets);
    }
    // Get the right child weight buffer for a node
    [[nodiscard]] XGBOOST_DEVICE common::Span<float> Right(bst_node_t nidx) const {
      return weights.subspan(nidx * n_targets * kWeightsPerNode + n_targets * 2, n_targets);
    }
  };

 private:
  // Persistent buffer for node weights, indexed by node id.
  dh::DeviceUVector<float> node_weights_;
  // Buffer for histogram scans.
  dh::DeviceUVector<GradientPairInt64> scan_buffer_;
  // Buffer for node gradient sums.
  dh::device_vector<GradientPairInt64> node_sums_;
  // Buffer for split sums (child_sum at split point), indexed by node id.
  dh::DeviceUVector<GradientPairInt64> split_sums_;

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
   *
   * @param max_nidx Maximum node ID among the nodes being evaluated. Used to allocate
   *                 weight storage on demand.
   */
  void EvaluateSplits(Context const *ctx, common::Span<MultiEvaluateSplitInputs const> d_inputs,
                      MultiEvaluateSplitSharedInputs const &shared_inputs, bst_node_t max_nidx,
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

  /**
   * @brief Allocate storage for split sums up to the given node ID.
   */
  void AllocSplitSum(bst_node_t nidx, bst_target_t n_targets) {
    auto end = (nidx + 1) * n_targets;
    if (this->split_sums_.size() < end) {
      this->split_sums_.resize(end);
    }
  }
  /**
   * @brief Get the split sum (child_sum at split point) for a node.
   */
  [[nodiscard]] common::Span<GradientPairInt64> GetSplitSum(bst_node_t nidx,
                                                            bst_target_t n_targets) {
    return GetNodeSumImpl(dh::ToSpan(this->split_sums_), nidx, n_targets);
  }
  [[nodiscard]] common::Span<GradientPairInt64 const> GetSplitSum(bst_node_t nidx,
                                                                  bst_target_t n_targets) const {
    return GetNodeSumImpl(dh::ToSpan(this->split_sums_), nidx, n_targets);
  }

  /**
   * @brief Ensure weight storage is allocated for a given node.
   *
   * This resizes the buffer on demand to accommodate the node ID.
   * Also tracks the split targets count which may differ from tree targets when using reduced
   * gradient.
   */
  void AllocNodeWeight(bst_node_t nidx, bst_target_t n_targets) {
    auto required = (nidx + 1) * n_targets * NodeWeightBuffer::kWeightsPerNode;
    if (this->node_weights_.size() < required) {
      this->node_weights_.resize(required);
    }
  }
  /**
   * @brief Get the weight buffer accessor for all nodes.
   *
   * Uses the split targets count stored during allocation, which may differ from tree targets
   * when using reduced gradient.
   */
  [[nodiscard]] NodeWeightBuffer GetNodeWeights(bst_target_t n_targets) {
    return NodeWeightBuffer{dh::ToSpan(this->node_weights_), n_targets};
  }
  /**
   * @brief Copy weights for a node from device to host vectors.
   *
   * Uses the split targets count stored during allocation, which may differ from tree targets
   * when using reduced gradient.
   *
   * TODO(jiamingy): Remove this method and use device-only buffer.
   */
  void CopyNodeWeightsToHost(bst_node_t nidx, bst_target_t n_targets,
                             std::vector<float> *base_weight, std::vector<float> *left_weight,
                             std::vector<float> *right_weight) {
    auto weights = this->GetNodeWeights(n_targets);
    base_weight->resize(n_targets);
    left_weight->resize(n_targets);
    right_weight->resize(n_targets);
    dh::CopyDeviceSpanToVector(base_weight, weights.Base(nidx));
    dh::CopyDeviceSpanToVector(left_weight, weights.Left(nidx));
    dh::CopyDeviceSpanToVector(right_weight, weights.Right(nidx));
  }

  // Track the child gradient sum.
  void ApplyTreeSplit(Context const *ctx, RegTree const *p_tree,
                      common::Span<MultiExpandEntry const> d_candidates, bst_target_t n_targets);
};
}  // namespace xgboost::tree::cuda_impl
