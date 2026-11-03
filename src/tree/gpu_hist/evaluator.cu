/**
 * Copyright 2022-2026, XGBoost Contributors
 *
 * @brief Some components of GPU Hist evaluator, this file only exist to reduce nvcc
 *        compilation time.
 */
#include <thrust/for_each.h>  // for for_each
#include <thrust/logical.h>   // for any_of
#include <thrust/sort.h>      // for stable_sort_by_key

#include <cuda/std/tuple>  // for make_tuple, get
#include <vector>          // for vector

#include "../../common/cuda_compat.cuh"   // for CUDA compatibility
#include "../../common/cuda_context.cuh"  // for CUDAContext
#include "../../common/device_helpers.cuh"
#include "../../common/hist_util.h"  // common::HistogramCuts
#include "evaluate_splits.cuh"
#include "xgboost/data.h"

namespace xgboost::tree {
namespace {
XGBOOST_DEVICE std::uint32_t CatBin(common::Span<std::uint32_t const> ptr,
                                    common::Span<std::uint32_t const> bin, std::uint32_t local) {
  auto seg = dh::SegmentId(ptr, local);
  return bin[seg] + static_cast<std::uint32_t>(local - ptr[seg]);
}
}  // namespace

void GPUHistEvaluator::Reset(Context const *ctx, common::HistogramCuts const &cuts,
                             common::Span<FeatureType const> ft, bst_feature_t n_features,
                             TrainParam const &param) {
  param_ = param;
  tree_evaluator_ = TreeEvaluator{param, n_features, ctx->Device(), 1u};
  has_categoricals_ = cuts.HasCategorical();
  cat_seg_bin_.clear();
  cat_seg_ptr_.clear();
  n_cat_bins_ = 0;
  if (cuts.HasCategorical()) {
    auto ptrs = cuts.cut_ptrs_.ConstDeviceSpan();
    auto beg = dh::make_counting_iterator<size_t>(1ul);
    auto end = dh::make_counting_iterator<size_t>(ptrs.size());
    auto to_onehot = param.max_cat_to_onehot;
    // This condition avoids sort-based split function calls if the users want
    // onehot-encoding-based splits.
    // For some reason, any_of adds 1.5 minutes to compilation time for CUDA 11.x.
    need_sort_histogram_ =
        thrust::any_of(ctx->CUDACtx()->CTP(), beg, end, [=] XGBOOST_DEVICE(size_t i) {
          auto idx = i - 1;
          if (common::IsCat(ft, idx)) {
            auto n_bins = ptrs[i] - ptrs[idx];
            bool use_sort = !common::UseOneHot(n_bins, to_onehot);
            return use_sort;
          }
          return false;
        });

    auto const &host_ptrs = cuts.Ptrs();
    CHECK_EQ(ft.size(), n_features);
    std::vector<FeatureType> host_ft(static_cast<std::size_t>(n_features));
    if (n_features > 0) {
      dh::safe_cuda(cudaMemcpyAsync(host_ft.data(), ft.data(), sizeof(FeatureType) * n_features,
                                    cudaMemcpyDefault, ctx->CUDACtx()->Stream()));
      dh::safe_cuda(cudaStreamSynchronize(ctx->CUDACtx()->Stream()));
    }
    std::vector<std::uint32_t> seg_ptr{0};
    std::vector<std::uint32_t> seg_bin;
    auto host_ft_span = common::Span<FeatureType const>{host_ft.data(), host_ft.size()};
    for (bst_feature_t f = 0; f < n_features; ++f) {
      if (!common::IsCat(host_ft_span, f)) {
        continue;
      }
      auto n_bins = host_ptrs[f + 1] - host_ptrs[f];
      if (n_bins == 0 || common::UseOneHot(n_bins, param.max_cat_to_onehot)) {
        continue;
      }
      seg_bin.push_back(static_cast<std::uint32_t>(host_ptrs[f]));
      seg_ptr.push_back(seg_ptr.back() + static_cast<std::uint32_t>(n_bins));
    }
    if (!seg_bin.empty()) {
      n_cat_bins_ = seg_ptr.back();
      cat_seg_ptr_.resize(seg_ptr.size());
      cat_seg_bin_.resize(seg_bin.size());
      dh::safe_cuda(cudaMemcpyAsync(cat_seg_ptr_.data().get(), seg_ptr.data(),
                                    seg_ptr.size() * sizeof(std::uint32_t), cudaMemcpyDefault,
                                    ctx->CUDACtx()->Stream()));
      dh::safe_cuda(cudaMemcpyAsync(cat_seg_bin_.data().get(), seg_bin.data(),
                                    seg_bin.size() * sizeof(std::uint32_t), cudaMemcpyDefault,
                                    ctx->CUDACtx()->Stream()));
    }

    node_categorical_storage_size_ =
        common::CatBitField::ComputeStorageSize(cuts.MaxCategory() + 1);
    CHECK_NE(node_categorical_storage_size_, 0);
    split_cats_.resize(node_categorical_storage_size_);
    h_split_cats_.resize(node_categorical_storage_size_);
    dh::safe_cuda(cudaMemsetAsync(split_cats_.data().get(), '\0',
                                  split_cats_.size() * sizeof(CatST), ctx->CUDACtx()->Stream()));

    cat_sorted_idx_.resize(cuts.cut_values_.Size() * 2);  // evaluate 2 nodes at a time.
    sort_input_.resize(cat_sorted_idx_.size());

    /**
     * cache feature index binary search result
     */
    feature_idx_.resize(cat_sorted_idx_.size());
    auto it = dh::make_counting_iterator(0ul);
    thrust::transform(ctx->CUDACtx()->CTP(), it, it + feature_idx_.size(), feature_idx_.begin(),
                      [=] XGBOOST_DEVICE(size_t i) {
                        auto fidx = dh::SegmentId(ptrs, i);
                        return fidx;
                      });
  }
  device_ = ctx->Device();
}

common::Span<bst_feature_t const> GPUHistEvaluator::SortHistogram(
    Context const *ctx, common::Span<const EvaluateSplitInputs> d_inputs,
    EvaluateSplitSharedInputs shared_inputs, TreeEvaluator::SplitEvaluator<EvalParam> evaluator) {
  auto sorted_idx = this->SortedIdx(d_inputs.size(), shared_inputs.feature_values.size());
  dh::Iota(sorted_idx, ctx->CUDACtx()->Stream());
  auto data = this->SortInput(d_inputs.size(), shared_inputs.feature_values.size());
  auto it = dh::make_counting_iterator(0u);
  auto d_feature_idx = dh::ToSpan(feature_idx_);
  auto total_bins = shared_inputs.feature_values.size();
  auto const num_nodes = static_cast<std::uint32_t>(d_inputs.size());
  auto seg_ptr = dh::ToSpan(cat_seg_ptr_);
  auto seg_bin = dh::ToSpan(cat_seg_bin_);
  auto const cat_bins = n_cat_bins_;
  auto const compact_n = static_cast<std::size_t>(num_nodes) * cat_bins;
  auto compact_key = data.subspan(0, compact_n);
  cat_compact_val_.resize(compact_n);
  auto compact_val = dh::ToSpan(cat_compact_val_);
  thrust::for_each(ctx->CUDACtx()->CTP(), it, it + compact_n, [=] XGBOOST_DEVICE(std::uint32_t k) {
    auto node = k / cat_bins;
    auto bin = CatBin(seg_ptr, seg_bin, k % cat_bins);
    auto global_idx = node * static_cast<std::uint32_t>(total_bins) + bin;
    auto const &input = d_inputs[node];
    auto grad = shared_inputs.rounding.ToFloatingPoint(input.gradient_histogram[bin]);
    auto lw = evaluator.CalcWeightCat(shared_inputs.param, grad);
    compact_key[k] = cuda::std::make_tuple(global_idx, lw);
    compact_val[k] = sorted_idx[global_idx];
  });
  // Sort an array segmented according to
  // - nodes
  // - features within each node
  // - gradients within each feature
  thrust::stable_sort_by_key(ctx->CUDACtx()->CTP(), dh::tbegin(compact_key), dh::tend(compact_key),
                             dh::tbegin(compact_val),
                             [=] XGBOOST_DEVICE(SortPair const &l, SortPair const &r) {
                               auto li = cuda::std::get<0>(l);
                               auto ri = cuda::std::get<0>(r);

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
                                 auto lw = cuda::std::get<1>(l);
                                 auto rw = cuda::std::get<1>(r);
                                 return lw < rw;
                               }
                               return li < ri;
                             });
  thrust::for_each(ctx->CUDACtx()->CTP(), it, it + compact_n, [=] XGBOOST_DEVICE(std::uint32_t k) {
    auto node = k / cat_bins;
    auto dst =
        node * static_cast<std::uint32_t>(total_bins) + CatBin(seg_ptr, seg_bin, k % cat_bins);
    sorted_idx[dst] = compact_val[k];
  });

  return dh::ToSpan(cat_sorted_idx_);
}
}  // namespace xgboost::tree
