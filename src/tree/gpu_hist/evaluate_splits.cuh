/*!
 * Copyright 2020 by XGBoost Contributors
 */
#ifndef EVALUATE_SPLITS_CUH_
#define EVALUATE_SPLITS_CUH_
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <xgboost/span.h>

#include "../../common/categorical.h"
#include "../split_evaluator.h"
#include "../updater_gpu_common.cuh"
#include "expand_entry.cuh"

namespace xgboost {
namespace common {
class HistogramCuts;
}

namespace tree {

// Inputs specific to each node
struct EvaluateSplitInputs {
  int nidx;
  GradientPairPrecise parent_sum;
  common::Span<const bst_feature_t> feature_set;
  common::Span<const GradientPairPrecise> gradient_histogram;
};

// Inputs necessary for all nodes
struct EvaluateSplitSharedInputs {
  GPUTrainingParam param;
  common::Span<FeatureType const> feature_types;
  common::Span<const uint32_t> feature_segments;
  common::Span<const float> feature_values;
  common::Span<const float> min_fvalue;
  XGBOOST_DEVICE auto Features() const { return feature_segments.size() - 1; }
  __device__ auto FeatureBins(bst_feature_t fidx) const {
    return feature_segments[fidx + 1] - feature_segments[fidx];
  }
};

template <typename GradientSumT>
class GPUHistEvaluator {
  using CatST = common::CatBitField::value_type;  // categorical storage type
  // use pinned memory to stage the categories, used for sort based splits.
  using Alloc = thrust::system::cuda::experimental::pinned_allocator<CatST>;

 private:
  TreeEvaluator tree_evaluator_;
  // storage for categories for each node, used for sort based splits.
  dh::device_vector<CatST> split_cats_;
  // host storage for categories for each node, used for sort based splits.
  std::vector<CatST, Alloc> h_split_cats_;
  // stream for copying categories from device back to host for expanding the decision tree.
  dh::CUDAStream copy_stream_;
  // storage for sorted index of feature histogram, used for sort based splits.
  dh::device_vector<bst_feature_t> cat_sorted_idx_;
  // cached input for sorting the histogram, used for sort based splits.
  using SortPair = thrust::tuple<uint32_t, double>;
  dh::device_vector<SortPair> sort_input_;
  // cache for feature index
  dh::device_vector<bst_feature_t> feature_idx_;
  // Training param used for evaluation
  TrainParam param_;
  // Do we have any categorical features that require sorting histograms?
  // use this to skip the expensive sort step
  bool need_sort_histogram_ = false;
  // Number of elements of categorical storage type
  // needed to hold categoricals for a single mode
  std::size_t node_categorical_storage_size_ = 0;

  // Copy the categories from device to host asynchronously.
  void CopyToHost(EvaluateSplitInputs const &input, common::Span<CatST> cats_out);

  /**
   * \brief Get host category storage of nidx for internal calculation.
   */
  auto HostCatStorage(bst_node_t nidx) {

    std::size_t min_size=(nidx+2)*node_categorical_storage_size_;
    if(h_split_cats_.size()<min_size){
      h_split_cats_.resize(min_size);
    }

    if (nidx == RegTree::kRoot) {
      auto cats_out = common::Span<CatST>{h_split_cats_}.subspan(nidx * node_categorical_storage_size_, node_categorical_storage_size_);
      return cats_out;
    }
    auto cats_out = common::Span<CatST>{h_split_cats_}.subspan(nidx * node_categorical_storage_size_, node_categorical_storage_size_ * 2);
    return cats_out;
  }

  /**
   * \brief Get device category storage of nidx for internal calculation.
   */
  auto DeviceCatStorage(bst_node_t nidx) {
    std::size_t min_size=(nidx+2)*node_categorical_storage_size_;
    if(split_cats_.size()<min_size){
      split_cats_.resize(min_size);
    }
    if (nidx == RegTree::kRoot) {
      auto cats_out = dh::ToSpan(split_cats_).subspan(nidx * node_categorical_storage_size_, node_categorical_storage_size_);
      return cats_out;
    }
    auto cats_out = dh::ToSpan(split_cats_).subspan(nidx * node_categorical_storage_size_, node_categorical_storage_size_ * 2);
    return cats_out;
  }

  /**
   * \brief Get sorted index storage based on the left node of inputs.
   */
  auto SortedIdx(EvaluateSplitInputs left, EvaluateSplitSharedInputs shared_inputs) {
    if (left.nidx == RegTree::kRoot && !cat_sorted_idx_.empty()) {
      return dh::ToSpan(cat_sorted_idx_).first(shared_inputs.feature_values.size());
    }
    return dh::ToSpan(cat_sorted_idx_);
  }

  auto SortInput(EvaluateSplitInputs left, EvaluateSplitSharedInputs shared_inputs) {
    if (left.nidx == RegTree::kRoot && !cat_sorted_idx_.empty()) {
      return dh::ToSpan(sort_input_).first(shared_inputs.feature_values.size());
    }
    return dh::ToSpan(sort_input_);
  }

 public:
  GPUHistEvaluator(TrainParam const &param, bst_feature_t n_features, int32_t device)
      : tree_evaluator_{param, n_features, device}, param_{param} {}
  /**
   * \brief Reset the evaluator, should be called before any use.
   */
  void Reset(common::HistogramCuts const &cuts, common::Span<FeatureType const> ft,
             bst_feature_t n_features, TrainParam const &param, int32_t device);

  /**
   * \brief Get host category storage for nidx.  Different from the internal version, this
   *        returns strictly 1 node.
   */
  common::Span<CatST const> GetHostNodeCats(bst_node_t nidx) const {
    copy_stream_.View().Sync();
    auto cats_out = common::Span<CatST const>{h_split_cats_}.subspan(nidx * node_categorical_storage_size_, node_categorical_storage_size_);
    return cats_out;
  }
  /**
   * \brief Add a split to the internal tree evaluator.
   */
  void ApplyTreeSplit(GPUExpandEntry const &candidate, RegTree *p_tree) {
    auto &tree = *p_tree;
    // Set up child constraints
    auto left_child = tree[candidate.nid].LeftChild();
    auto right_child = tree[candidate.nid].RightChild();
    tree_evaluator_.AddSplit(candidate.nid, left_child, right_child,
                             tree[candidate.nid].SplitIndex(), candidate.left_weight,
                             candidate.right_weight);
  }

  auto GetEvaluator() { return tree_evaluator_.GetEvaluator<GPUTrainingParam>(); }
  /**
   * \brief Sort the histogram based on output to obtain contiguous partitions.
   */
  common::Span<bst_feature_t const> SortHistogram(
      EvaluateSplitInputs const &left, EvaluateSplitInputs const &right,EvaluateSplitSharedInputs shared_inputs,
      TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator);

  // impl of evaluate splits, contains CUDA kernels so it's public
  void LaunchEvaluateSplits(EvaluateSplitInputs left,
                      EvaluateSplitInputs right,EvaluateSplitSharedInputs shared_inputs, 
                      TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
                      common::Span<DeviceSplitCandidate> out_splits);
  /**
   * \brief Evaluate splits for left and right nodes.
   */
  void EvaluateSplits(GPUExpandEntry candidate,
                      EvaluateSplitInputs left,
                      EvaluateSplitInputs right,EvaluateSplitSharedInputs shared_inputs, 
                      common::Span<GPUExpandEntry> out_splits);
  /**
   * \brief Evaluate splits for root node.
   */
  GPUExpandEntry EvaluateSingleSplit(EvaluateSplitInputs input,EvaluateSplitSharedInputs shared_inputs, float weight);
};
}  // namespace tree
}  // namespace xgboost

#endif  // EVALUATE_SPLITS_CUH_
