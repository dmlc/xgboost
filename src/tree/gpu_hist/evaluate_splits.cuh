/**
 * Copyright 2020-2026, XGBoost Contributors
 *
 * @brief Split evaluation shared by the scalar and the vector leaf GPU trees.
 */
#ifndef EVALUATE_SPLITS_CUH_
#define EVALUATE_SPLITS_CUH_

#include <algorithm>  // for max
#include <cstddef>    // for size_t
#include <cstdint>    // for uint32_t
#include <memory>     // for unique_ptr
#include <vector>     // for vector

#include "../../common/categorical.h"           // for IsCat
#include "../../common/cuda_pinned_allocator.h"  // for PinnedAllocator
#include "../../common/device_helpers.cuh"      // for PinnedMemory
#include "../../common/device_vector.cuh"       // for DeviceUVector, CachingDeviceUVector
#include "../split_evaluator.h"                 // for TreeEvaluator, EvalParam
#include "../updater_gpu_common.cuh"            // for DeviceSplitCandidate
#include "expand_entry.cuh"                     // for GPUExpandEntry
#include "quantiser.cuh"                        // for GradientQuantiser
#include "xgboost/base.h"                       // for GradientPairInt64
#include "xgboost/context.h"                    // for Context
#include "xgboost/span.h"                       // for Span

namespace xgboost {
class RegTree;

namespace tree {
// Input for evaluation kernel for each tree node.
struct EvaluateSplitInputs {
  bst_node_t nidx;
  bst_node_t depth;
  // len == n_targets
  common::Span<GradientPairInt64 const> parent_sum;
  common::Span<bst_feature_t const> feature_set;
  // Shape: [n_targets, n_total_bins_per_tar]
  common::Span<GradientPairInt64 const> histogram;
};

// Input for evaluation kernel that can be shared by multiple tree nodes.
struct EvaluateSplitSharedInputs {
  // len == n_targets
  common::Span<GradientQuantiser const> roundings;
  // cut pointers
  common::Span<std::uint32_t const> feature_segments;
  // cut values
  float const* feature_values;
  common::Span<FeatureType const> feature_types;
  // Number of bit-field storage elements required for one categorical split.
  std::size_t cat_storage_size{0};
  // Number of histogram bins for one target, across all features.
  bst_bin_t n_total_bins_per_tar;
  bst_feature_t max_active_feature;
  EvalParam param;

  [[nodiscard]] XGBOOST_DEVICE bst_target_t Targets() const { return roundings.size(); }
  [[nodiscard]] XGBOOST_DEVICE bst_feature_t Features() const {
    return this->feature_segments.size() - 1;
  }
  [[nodiscard]] XGBOOST_DEVICE bool IsCategorical(bst_feature_t fidx) const {
    return common::IsCat(this->feature_types, fidx);
  }
  [[nodiscard]] XGBOOST_DEVICE bst_bin_t FeatureBins(bst_feature_t fidx) const {
    return this->feature_segments[fidx + 1] - this->feature_segments[fidx];
  }
  // Whether the feature uses a sort-based partition split.
  [[nodiscard]] XGBOOST_DEVICE bool IsPartition(bst_feature_t fidx) const {
    return this->IsCategorical(fidx) &&
           !common::UseOneHot(this->FeatureBins(fidx), this->param.max_cat_to_onehot);
  }
};

/** @brief Evaluator for both scalar and vector leaves. */
class GPUHistEvaluator {
 public:
  // Split applied to the tree, with the IDs of the new children.
  struct NodeSplit {
    bst_node_t nidx;
    bst_node_t left_nidx;
    bst_node_t right_nidx;
    bst_feature_t fidx;
    DefaultDirection dir;
  };

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
  using CatAlloc = common::cuda_impl::PinnedAllocator<CatWordT>;

  std::unique_ptr<TreeEvaluator> tree_evaluator_{nullptr};
  bst_target_t n_targets_{0};
  // The caller synchronizes the stream after ApplyTreeSplit before the buffers are reused.
  dh::PinnedMemory h_splits_;
  dh::CachingDeviceUVector<NodeSplit> d_splits_;
  // Persistent buffer for node weights, indexed by node id.
  dh::DeviceUVector<float> node_weights_;
  // Child sums of the best split for each (node, feature), also used as the scan carry.
  dh::DeviceUVector<GradientPairInt64> scratch_;
  // Buffer for node gradient sums. Nodes stored in this buffer are valid nodes (exist in
  // the output tree) instead of candidates. Shape: [n_nodes, n_targets]
  dh::DeviceUVector<GradientPairInt64> node_sums_;
  // Buffer for split sums (child_sum at split point), indexed by node id. This temporary
  // buffer is needed because we don't have the child node index during evaluation, which
  // is only available after applying split to the tree.
  dh::DeviceUVector<GradientPairInt64> split_sums_;
  // Category bit fields for evaluated nodes, indexed by node id. They must remain valid
  // while candidates are waiting in the loss-guide queue.
  dh::DeviceUVector<CatWordT> split_cats_;
  std::size_t node_cat_storage_size_{0};
  // Host copies of node weights and categories for the scalar tree.
  std::vector<float> h_node_weights_;
  std::vector<CatWordT, CatAlloc> h_split_cats_;
  // Whether any categorical feature requires partition-based evaluation.
  bool need_sort_histogram_{false};
  // Node-local bin indices sorted by the category score, used by partition-based splits.
  dh::DeviceUVector<bst_bin_t> sorted_idx_;
  // Cached feature index for each bin, used by partition-based splits.
  dh::DeviceUVector<bst_feature_t> feature_idx_;

  // Grow a buffer indexed by node ID to hold up to max_nidx. The size is at least doubled to
  // amortize the allocation for the loss-guide policy, which adds a few nodes at a time.
  template <typename T>
  static void Grow(dh::DeviceUVector<T>* vec, bst_node_t max_nidx, std::size_t n_per_node) {
    auto required = static_cast<std::size_t>(max_nidx + 1) * n_per_node;
    if (vec->size() < required) {
      vec->resize(std::max(required, vec->size() * 2));
    }
  }

 public:
  // Sort the histogram based on output to obtain contiguous partitions.
  common::Span<bst_bin_t const> SortHistogram(Context const* ctx,
                                              common::Span<EvaluateSplitInputs const> d_inputs,
                                              EvaluateSplitSharedInputs const& shared_inputs);
  void Reset(Context const* ctx, common::Span<std::uint32_t const> feature_segments,
             common::Span<FeatureType const> feature_types, TrainParam const& param,
             bst_target_t n_targets);

  [[nodiscard]] auto GetEvaluator() const { return tree_evaluator_->GetEvaluator<EvalParam>(); }

  /**
   * @brief Run evaluation for the root node.
   */
  [[nodiscard]] GPUExpandEntry EvaluateSingleSplit(Context const* ctx,
                                                   EvaluateSplitInputs const& input,
                                                   EvaluateSplitSharedInputs const& shared_inputs);
  /**
   * @brief Run evaluation for multiple nodes.
   *
   * @param max_nidx Maximum node ID among the nodes being evaluated. Used to allocate
   *                 weight storage on demand.
   */
  void EvaluateSplits(Context const* ctx, common::Span<EvaluateSplitInputs const> d_inputs,
                      EvaluateSplitSharedInputs const& shared_inputs, bst_node_t max_nidx,
                      common::Span<GPUExpandEntry> out_splits);

  /**
   * @brief Allocate storage for node sums up to the given node ID.
   */
  void AllocNodeSum(bst_node_t nidx) { Grow(&this->node_sums_, nidx, this->n_targets_); }
  [[nodiscard]] common::Span<GradientPairInt64> GetNodeSum(bst_node_t nidx) {
    return dh::ToSpan(this->node_sums_).subspan(nidx * this->n_targets_, this->n_targets_);
  }
  [[nodiscard]] NodeWeightBuffer GetNodeWeights() {
    return NodeWeightBuffer{dh::ToSpan(this->node_weights_), this->n_targets_};
  }
  [[nodiscard]] common::Span<CatWordT const> GetNodeCats(bst_node_t nidx) const {
    return dh::ToSpan(this->split_cats_)
        .subspan(nidx * this->node_cat_storage_size_, this->node_cat_storage_size_);
  }

  /**
   * @brief Copy node weights and categories in [first_nidx, last_nidx] to the host. The
   *        copy is ordered on the CUDA stream of the context.
   */
  void CopyToHost(Context const* ctx, bst_node_t first_nidx, bst_node_t last_nidx);
  [[nodiscard]] NodeWeightBuffer GetHostNodeWeights() {
    common::Span<float> weights{this->h_node_weights_.data(), this->h_node_weights_.size()};
    return NodeWeightBuffer{weights, this->n_targets_};
  }
  [[nodiscard]] common::Span<CatWordT const> GetHostNodeCats(bst_node_t nidx) const {
    return common::Span{this->h_split_cats_.data(), this->h_split_cats_.size()}.subspan(
        nidx * this->node_cat_storage_size_, this->node_cat_storage_size_);
  }

  // Update the tree evaluator state and track child gradient sums after the tree expansion.
  void ApplyTreeSplit(Context const* ctx, RegTree const* p_tree,
                      common::Span<GPUExpandEntry const> h_candidates);
};
}  // namespace tree
}  // namespace xgboost

#endif  // EVALUATE_SPLITS_CUH_
