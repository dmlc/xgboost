/*!
 * Copyright 2022 by XGBoost Contributors
 *
 * \brief Some components of GPU Hist evaluator, this file only exist to reduce nvcc
 *        compilation time.
 */
#include <thrust/logical.h>
#include <thrust/sort.h>

#include "../../common/hist_util.h"  // common::HistogramCuts
#include "../../data/ellpack_page.cuh"
#include "evaluate_splits.cuh"
#include "xgboost/data.h"

namespace xgboost {
namespace tree {
namespace {
struct UseSortOp {
  uint32_t to_onehot;
  common::Span<uint32_t const> ptrs;
  common::Span<FeatureType const> ft;
  ObjInfo task;

  XGBOOST_DEVICE bool operator()(size_t i) {
    auto idx = i - 1;
    if (common::IsCat(ft, idx)) {
      auto n_bins = ptrs[i] - ptrs[idx];
      bool use_sort = !common::UseOneHot(n_bins, to_onehot, task);
      return use_sort;
    }
    return false;
  }
};
}  // anonymous namespace

template <typename GradientSumT>
void GPUHistEvaluator<GradientSumT>::Reset(common::HistogramCuts const &cuts,
                                           common::Span<FeatureType const> ft, ObjInfo task,
                                           bst_feature_t n_features, TrainParam const &param,
                                           int32_t device) {
  param_ = param;
  tree_evaluator_ = TreeEvaluator{param, n_features, device};
  if (cuts.HasCategorical() && !task.UseOneHot()) {
    auto ptrs = cuts.cut_ptrs_.ConstDeviceSpan();
    auto beg = thrust::make_counting_iterator(1ul);
    auto end = thrust::make_counting_iterator(ptrs.size());
    auto to_onehot = param.max_cat_to_onehot;
    // for some reason, any_of adds 1.5 minutes to compilation time for CUDA 11.x
    has_sort_ = thrust::any_of(thrust::device, beg, end, UseSortOp{to_onehot, ptrs, ft, task});

    if (has_sort_) {
      auto bit_storage_size = common::CatBitField::ComputeStorageSize(cuts.MaxCategory() + 1);
      CHECK_NE(bit_storage_size, 0);
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
  dh::XGBDeviceAllocator<char> alloc;
  auto sorted_idx = this->SortedIdx(left);
  dh::Iota(sorted_idx);
  // sort 2 nodes and all the features at the same time, disregarding colmun sampling.
  thrust::stable_sort(
      thrust::cuda::par(alloc), dh::tbegin(sorted_idx), dh::tend(sorted_idx),
      [evaluator, left, right] XGBOOST_DEVICE(size_t l, size_t r) {
        auto l_is_left = l < left.feature_values.size();
        auto r_is_left = r < left.feature_values.size();
        if (l_is_left != r_is_left) {
          return l_is_left;  // not the same node
        }

        auto const &input = l_is_left ? left : right;
        l -= (l_is_left ? 0 : input.feature_values.size());
        r -= (r_is_left ? 0 : input.feature_values.size());

        auto lfidx = dh::SegmentId(input.feature_segments, l);
        auto rfidx = dh::SegmentId(input.feature_segments, r);
        if (lfidx != rfidx) {
          return lfidx < rfidx;  // not the same feature
        }
        if (common::IsCat(input.feature_types, lfidx)) {
          auto lw = evaluator.CalcWeightCat(input.param, input.gradient_histogram[l]);
          auto rw = evaluator.CalcWeightCat(input.param, input.gradient_histogram[r]);
          return lw < rw;
        }
        return l < r;
      });
  return dh::ToSpan(cat_sorted_idx_);
}

template class GPUHistEvaluator<GradientPair>;
template class GPUHistEvaluator<GradientPairPrecise>;
}  // namespace tree
}  // namespace xgboost
