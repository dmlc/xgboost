/*!
 * Copyright 2020 by XGBoost Contributors
 */
#ifndef EVALUATE_SPLITS_CUH_
#define EVALUATE_SPLITS_CUH_
#include <xgboost/span.h>

#include "../../common/categorical.h"
#include "../../common/cuda_pinned_allocator.h"
#include "../split_evaluator.h"
#include "../updater_gpu_common.cuh"
#include "expand_entry.cuh"
#include "histogram.cuh"

namespace xgboost {
namespace common {
class HistogramCuts;
}

namespace tree {

// Inputs specific to each node
struct EvaluateSplitInputs {
  int nidx;
  int depth;
  GradientPairInt64 parent_sum;
  common::Span<const bst_feature_t> feature_set;
  common::Span<const GradientPairInt64> gradient_histogram;
};

// Inputs necessary for all nodes
struct EvaluateSplitSharedInputs {
  GPUTrainingParam param;
  GradientQuantiser rounding;
  common::Span<FeatureType const> feature_types;
  common::Span<const uint32_t> feature_segments;
  common::Span<const float> feature_values;
  common::Span<const float> min_fvalue;
  bool is_dense;
  XGBOOST_DEVICE auto Features() const { return feature_segments.size() - 1; }
  __device__ auto FeatureBins(bst_feature_t fidx) const {
    return feature_segments[fidx + 1] - feature_segments[fidx];
  }
};

// Used to return internal storage regions for categoricals
// Usable on device
struct CatAccessor {
  common::Span<common::CatBitField::value_type> cat_storage;
  std::size_t node_categorical_storage_size;
  XGBOOST_DEVICE common::Span<common::CatBitField::value_type> GetNodeCatStorage(bst_node_t nidx) {
    return this->cat_storage.subspan(nidx * this->node_categorical_storage_size,
                                     this->node_categorical_storage_size);
  }
};

class GPUHistEvaluator {
  using CatST = common::CatBitField::value_type;  // categorical storage type
  // use pinned memory to stage the categories, used for sort based splits.
  using Alloc = xgboost::common::cuda::pinned_allocator<CatST>;

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
  using SortPair = thrust::tuple<uint32_t, float>;
  dh::device_vector<SortPair> sort_input_;
  // cache for feature index
  dh::device_vector<bst_feature_t> feature_idx_;
  // Training param used for evaluation
  TrainParam param_;
  // Do we have any categorical features that require sorting histograms?
  // use this to skip the expensive sort step
  bool need_sort_histogram_ = false;
  bool has_categoricals_ = false;
  // Number of elements of categorical storage type
  // needed to hold categoricals for a single mode
  std::size_t node_categorical_storage_size_ = 0;

  // Copy the categories from device to host asynchronously.
  void CopyToHost( const std::vector<bst_node_t>& nidx);

  /**
   * \brief Get host category storage of nidx for internal calculation.
   */
  auto HostCatStorage(const std::vector<bst_node_t> &nidx) {
    if (!has_categoricals_) return CatAccessor{};
    auto max_nidx = *std::max_element(nidx.begin(), nidx.end());
    std::size_t min_size = (max_nidx + 2) * node_categorical_storage_size_;
    if (h_split_cats_.size() < min_size) {
      h_split_cats_.resize(min_size);
    }
    return CatAccessor{{h_split_cats_.data(), h_split_cats_.size()},
                       node_categorical_storage_size_};
  }

  /**
   * \brief Get device category storage of nidx for internal calculation.
   */
  auto DeviceCatStorage(const std::vector<bst_node_t> &nidx) {
    if (!has_categoricals_) return CatAccessor{};
    auto max_nidx = *std::max_element(nidx.begin(), nidx.end());
    std::size_t min_size = (max_nidx + 2) * node_categorical_storage_size_;
    if (split_cats_.size() < min_size) {
      split_cats_.resize(min_size);
    }
    return CatAccessor{dh::ToSpan(split_cats_), node_categorical_storage_size_};
  }

  /**
   * \brief Get sorted index storage based on the left node of inputs.
   */
  auto SortedIdx(int num_nodes, bst_feature_t total_bins) {
    if(!need_sort_histogram_) return common::Span<bst_feature_t>();
    cat_sorted_idx_.resize(num_nodes * total_bins);
    return dh::ToSpan(cat_sorted_idx_);
  }

  auto SortInput(int num_nodes, bst_feature_t total_bins) {
    if(!need_sort_histogram_) return common::Span<SortPair>();
    sort_input_.resize(num_nodes * total_bins);
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
    auto cats_out = common::Span<CatST const>{h_split_cats_}.subspan(
        nidx * node_categorical_storage_size_, node_categorical_storage_size_);
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
  common::Span<bst_feature_t const> SortHistogram(common::Span<const EvaluateSplitInputs> d_inputs,
      EvaluateSplitSharedInputs shared_inputs,
      TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator);

  // impl of evaluate splits, contains CUDA kernels so it's public
  void LaunchEvaluateSplits(
      bst_feature_t max_active_features,
      common::Span<const EvaluateSplitInputs> d_inputs,
      EvaluateSplitSharedInputs shared_inputs,
      TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
      common::Span<DeviceSplitCandidate> out_splits);
  /**
   * \brief Evaluate splits for left and right nodes.
   */
  void EvaluateSplits(const std::vector<bst_node_t> &nidx,
                      bst_feature_t max_active_features,
                      common::Span<const EvaluateSplitInputs> d_inputs,
                      EvaluateSplitSharedInputs shared_inputs,
                      common::Span<GPUExpandEntry> out_splits);
  /**
   * \brief Evaluate splits for root node.
   */
  GPUExpandEntry EvaluateSingleSplit(EvaluateSplitInputs input,
                                     EvaluateSplitSharedInputs shared_inputs);
};
}  // namespace tree
}  // namespace xgboost

#endif  // EVALUATE_SPLITS_CUH_
