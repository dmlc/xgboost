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
  template <typename GradT>
  static XGBOOST_DEVICE common::Span<GradT> GetNodeSumImpl(common::Span<GradT> node_sums,
                                                           bst_node_t nidx,
                                                           bst_target_t n_targets) {
    auto offset = nidx * n_targets;
    return node_sums.subspan(offset, n_targets);
  }

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

  struct NodeSumBuffer {
    dh::DeviceUVector<GradientPairInt64> node_sums;

    /**
     * @brief Allocate storage for split sums up to the given node ID.
     */
    void Alloc(bst_node_t nidx, bst_target_t n_targets) {
      auto end = (nidx + 1) * n_targets;
      if (this->node_sums.size() < end) {
        this->node_sums.resize(end);
      }
    }
    [[nodiscard]] common::Span<GradientPairInt64> GetNode(bst_node_t nidx, bst_target_t n_targets) {
      return GetNodeSumImpl(dh::ToSpan(this->node_sums), nidx, n_targets);
    }
    [[nodiscard]] common::Span<GradientPairInt64 const> GetNode(bst_node_t nidx,
                                                                bst_target_t n_targets) const {
      return GetNodeSumImpl(dh::ToSpan(this->node_sums), nidx, n_targets);
    }
    auto View() { return dh::ToSpan(this->node_sums); }
    auto View() const { return dh::ToSpan(this->node_sums); }
  };

 private:
  // Persistent buffer for node weights, indexed by node id.
  dh::DeviceUVector<float> node_weights_;
  // Buffer for histogram scans.
  dh::DeviceUVector<GradientPairInt64> scan_buffer_;
  // Buffer for node gradient sums. Nodes stored by this buffer are valid nodes (exist in
  // the output tree) instead of candidates.
  NodeSumBuffer node_sums_;
  // Buffer for split sums (child_sum at split point), indexed by node id. This temporary
  // buffer is needed because we don't have the child node index during evaluation, which
  // is only available after applying split to the tree.
  NodeSumBuffer split_sums_;

 public:
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

  /**
   * @brief Allocate storage for node sums up to the given node ID.
   */
  void AllocNodeSum(bst_node_t nidx, bst_target_t n_targets) {
    this->node_sums_.Alloc(nidx, n_targets);
  }
  [[nodiscard]] common::Span<GradientPairInt64> GetNodeSum(bst_node_t nidx,
                                                           bst_target_t n_targets) {
    return this->node_sums_.GetNode(nidx, n_targets);
  }

  /**
   * @brief Allocate storage for weights up to the given node ID.
   */
  void AllocNodeWeight(bst_node_t nidx, bst_target_t n_targets) {
    auto required = (nidx + 1) * n_targets * NodeWeightBuffer::kWeightsPerNode;
    if (this->node_weights_.size() < required) {
      this->node_weights_.resize(required);
    }
  }
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
