/*!
 * Copyright 2022 by XGBoost Contributors
 *
 * \brief Some components of GPU Hist evaluator, this file only exist to reduce nvcc
 *        compilation time.
 */
#include <thrust/logical.h>  // thrust::any_of
#include <thrust/sort.h>     // thrust::stable_sort

#include "../../common/device_helpers.cuh"
#include "../../common/hist_util.h"  // common::HistogramCuts
#include "evaluate_splits.cuh"
#include "xgboost/data.h"

namespace xgboost {
namespace tree {
void GPUHistEvaluator::Reset(common::HistogramCuts const &cuts,
                                           common::Span<FeatureType const> ft,
                                           bst_feature_t n_features, TrainParam const &param,
                                           int32_t device) {
  param_ = param;
  tree_evaluator_ = TreeEvaluator{param, n_features, device};
  has_categoricals_ = cuts.HasCategorical();
  if (cuts.HasCategorical()) {
    dh::XGBCachingDeviceAllocator<char> alloc;
    auto ptrs = cuts.cut_ptrs_.ConstDeviceSpan();
    auto beg = thrust::make_counting_iterator<size_t>(1ul);
    auto end = thrust::make_counting_iterator<size_t>(ptrs.size());
    auto to_onehot = param.max_cat_to_onehot;
    // This condition avoids sort-based split function calls if the users want
    // onehot-encoding-based splits.
    // For some reason, any_of adds 1.5 minutes to compilation time for CUDA 11.x.
    need_sort_histogram_ =
        thrust::any_of(thrust::cuda::par(alloc), beg, end, [=] XGBOOST_DEVICE(size_t i) {
          auto idx = i - 1;
          if (common::IsCat(ft, idx)) {
            auto n_bins = ptrs[i] - ptrs[idx];
            bool use_sort = !common::UseOneHot(n_bins, to_onehot);
            return use_sort;
          }
          return false;
        });

    node_categorical_storage_size_ =
        common::CatBitField::ComputeStorageSize(cuts.MaxCategory() + 1);
    CHECK_NE(node_categorical_storage_size_, 0);
    split_cats_.resize(node_categorical_storage_size_);
    h_split_cats_.resize(node_categorical_storage_size_);
    dh::safe_cuda(
        cudaMemsetAsync(split_cats_.data().get(), '\0', split_cats_.size() * sizeof(CatST)));

    cat_sorted_idx_.resize(cuts.cut_values_.Size() * 2);  // evaluate 2 nodes at a time.
    sort_input_.resize(cat_sorted_idx_.size());

    /**
     * cache feature index binary search result
     */
    feature_idx_.resize(cat_sorted_idx_.size());
    auto d_fidxes = dh::ToSpan(feature_idx_);
    auto it = thrust::make_counting_iterator(0ul);
    auto values = cuts.cut_values_.ConstDeviceSpan();
    thrust::transform(thrust::cuda::par(alloc), it, it + feature_idx_.size(), feature_idx_.begin(),
                      [=] XGBOOST_DEVICE(size_t i) {
                        auto fidx = dh::SegmentId(ptrs, i);
                        return fidx;
                      });
  }
}

common::Span<bst_feature_t const> GPUHistEvaluator::SortHistogram(
    common::Span<const EvaluateSplitInputs> d_inputs, EvaluateSplitSharedInputs shared_inputs,
    TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator) {
  dh::XGBCachingDeviceAllocator<char> alloc;
  auto sorted_idx = this->SortedIdx(d_inputs.size(), shared_inputs.feature_values.size());
  dh::Iota(sorted_idx);
  auto data = this->SortInput(d_inputs.size(), shared_inputs.feature_values.size());
  auto it = thrust::make_counting_iterator(0u);
  auto d_feature_idx = dh::ToSpan(feature_idx_);
  auto total_bins = shared_inputs.feature_values.size();
  thrust::transform(thrust::cuda::par(alloc), it, it + data.size(), dh::tbegin(data),
                    [=] XGBOOST_DEVICE(uint32_t i) {
                      auto const &input = d_inputs[i / total_bins];
                      auto j = i % total_bins;
                      auto fidx = d_feature_idx[j];
                      if (common::IsCat(shared_inputs.feature_types, fidx)) {
                        auto grad =
                            shared_inputs.rounding.ToFloatingPoint(input.gradient_histogram[j]);
                        auto lw = evaluator.CalcWeightCat(shared_inputs.param, grad);
                        return thrust::make_tuple(i, lw);
                      }
                      return thrust::make_tuple(i, 0.0f);
                    });
  // Sort an array segmented according to
  // - nodes
  // - features within each node
  // - gradients within each feature
  thrust::stable_sort_by_key(thrust::cuda::par(alloc), dh::tbegin(data), dh::tend(data),
                             dh::tbegin(sorted_idx),
                             [=] XGBOOST_DEVICE(SortPair const &l, SortPair const &r) {
                               auto li = thrust::get<0>(l);
                               auto ri = thrust::get<0>(r);

                               auto l_node = li / total_bins;
                               auto r_node = ri / total_bins;

                               if (l_node != r_node) {
                                 return l_node < r_node;  // not the same node
                               }

                               li = li % total_bins;
                               ri = ri % total_bins;

                               auto lfidx = d_feature_idx[li];
                               auto rfidx = d_feature_idx[ri];

                               if (lfidx != rfidx) {
                                 return lfidx < rfidx;  // not the same feature
                               }

                               if (common::IsCat(shared_inputs.feature_types, lfidx)) {
                                 auto lw = thrust::get<1>(l);
                                 auto rw = thrust::get<1>(r);
                                 return lw < rw;
                               }
                               return li < ri;
                             });
  return dh::ToSpan(cat_sorted_idx_);
}

}  // namespace tree
}  // namespace xgboost
