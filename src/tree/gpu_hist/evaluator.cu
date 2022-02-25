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
template <typename GradientSumT>
void GPUHistEvaluator<GradientSumT>::Reset(common::HistogramCuts const &cuts,
                                           common::Span<FeatureType const> ft, ObjInfo task,
                                           bst_feature_t n_features, TrainParam const &param,
                                           int32_t device) {
  param_ = param;
  tree_evaluator_ = TreeEvaluator{param, n_features, device};
  if (cuts.HasCategorical() && !task.UseOneHot()) {
    dh::XGBCachingDeviceAllocator<char> alloc;
    auto ptrs = cuts.cut_ptrs_.ConstDeviceSpan();
    auto beg = thrust::make_counting_iterator<size_t>(1ul);
    auto end = thrust::make_counting_iterator<size_t>(ptrs.size());
    auto to_onehot = param.max_cat_to_onehot;
    // This condition avoids sort-based split function calls if the users want
    // onehot-encoding-based splits.
    // For some reason, any_of adds 1.5 minutes to compilation time for CUDA 11.x.
    has_sort_ = thrust::any_of(thrust::cuda::par(alloc), beg, end, [=] XGBOOST_DEVICE(size_t i) {
      auto idx = i - 1;
      if (common::IsCat(ft, idx)) {
        auto n_bins = ptrs[i] - ptrs[idx];
        bool use_sort = !common::UseOneHot(n_bins, to_onehot, task);
        return use_sort;
      }
      return false;
    });

    if (has_sort_) {
      auto bit_storage_size = common::CatBitField::ComputeStorageSize(cuts.MaxCategory() + 1);
      CHECK_NE(bit_storage_size, 0);
      // We need to allocate for all nodes since the updater can grow the tree layer by
      // layer, all nodes in the same layer must be preserved until that layer is
      // finished.  We can allocate one layer at a time, but the best case is reducing the
      // size of the bitset by about a half, at the cost of invoking CUDA malloc many more
      // times than necessary.
      split_cats_.resize(param.MaxNodes() * bit_storage_size);
      h_split_cats_.resize(split_cats_.size());
      dh::safe_cuda(
          cudaMemsetAsync(split_cats_.data().get(), '\0', split_cats_.size() * sizeof(CatST)));

      cat_sorted_idx_.resize(cuts.cut_values_.Size() * 2);  // evaluate 2 nodes at a time.
    }
  }
}

template <typename GradientSumT>
common::Span<bst_feature_t const> GPUHistEvaluator<GradientSumT>::SortHistogram(
    EvaluateSplitInputs<GradientSumT> const &left, EvaluateSplitInputs<GradientSumT> const &right,
    TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator) {
  dh::XGBCachingDeviceAllocator<char> alloc;
  auto sorted_idx = this->SortedIdx(left);
  dh::Iota(sorted_idx);
  using Tuple = thrust::tuple<bst_feature_t, uint32_t, double>;
  auto it = dh::MakeTransformIterator<Tuple>(
      thrust::make_counting_iterator(0u), [=] XGBOOST_DEVICE(uint32_t i) {
        auto is_left = i < left.feature_values.size();
        auto const &input = is_left ? left : right;
        auto j = i - (is_left ? 0 : input.feature_values.size());
        auto fidx = dh::SegmentId(input.feature_segments, j);
        if (common::IsCat(input.feature_types, fidx)) {
          auto lw = evaluator.CalcWeightCat(input.param, input.gradient_histogram[j]);
          return thrust::make_tuple(fidx, i, lw);
        }
        return thrust::make_tuple(fidx, i, 0.0);
      });
  thrust::stable_sort_by_key(
      thrust::cuda::par(alloc), it, it + sorted_idx.size(), dh::tbegin(sorted_idx),
      [=] XGBOOST_DEVICE(Tuple const &l, Tuple const &r) {
        auto lfidx = thrust::get<0>(l);
        auto rfidx = thrust::get<0>(r);

        auto li = thrust::get<1>(l);
        auto ri = thrust::get<1>(r);

        auto l_is_left = li < left.feature_values.size();
        auto r_is_left = ri < left.feature_values.size();

        if (l_is_left != r_is_left) {
          return l_is_left;  // not the same node
        }
        if (lfidx != rfidx) {
          return lfidx < rfidx;  // not the same feature
        }

        auto const &input = l_is_left ? left : right;
        li -= (l_is_left ? 0 : input.feature_values.size());
        ri -= (r_is_left ? 0 : input.feature_values.size());
        if (common::IsCat(input.feature_types, lfidx)) {
          auto lw = evaluator.CalcWeightCat(input.param, input.gradient_histogram[li]);
          auto rw = evaluator.CalcWeightCat(input.param, input.gradient_histogram[ri]);
          return lw < rw;
        }
        return li < ri;
      });
  return dh::ToSpan(cat_sorted_idx_);
}

template class GPUHistEvaluator<GradientPair>;
template class GPUHistEvaluator<GradientPairPrecise>;
}  // namespace tree
}  // namespace xgboost
