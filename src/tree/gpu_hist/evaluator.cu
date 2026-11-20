/**
 * Copyright 2022-2026, XGBoost Contributors
 *
 * @brief Some components of GPU Hist evaluator, this file only exist to reduce nvcc
 *        compilation time.
 */
#include <thrust/copy.h>      // for copy_if
#include <thrust/for_each.h>  // for for_each
#include <thrust/logical.h>   // for any_of
#include <thrust/sort.h>      // for stable_sort

#include <cuda/std/tuple>  // for make_tuple, get, tuple
#include <memory>          // for make_unique

#include "../../common/cuda_compat.cuh"   // for CUDA compatibility
#include "../../common/cuda_context.cuh"  // for CUDAContext
#include "../../common/device_helpers.cuh"
#include "evaluate_splits.cuh"

namespace xgboost::tree {
void GPUHistEvaluator::Reset(Context const *ctx,
                             common::Span<std::uint32_t const> feature_segments,
                             common::Span<FeatureType const> feature_types,
                             TrainParam const &param, bst_target_t n_targets) {
  CHECK_GT(feature_segments.size(), 0);
  CHECK_GT(n_targets, 0);
  auto n_features = static_cast<bst_feature_t>(feature_segments.size() - 1);
  this->tree_evaluator_ =
      std::make_unique<TreeEvaluator>(param, n_features, ctx->Device(), n_targets);
  this->n_targets_ = n_targets;
  this->need_sort_histogram_ = false;
  // Invalidate the cached feature index, the cuts can change between trees.
  this->feature_idx_.resize(0);
  if (feature_types.empty()) {
    return;
  }

  auto feature_it = dh::make_counting_iterator<bst_feature_t>(0);
  auto max_cat_to_onehot = param.max_cat_to_onehot;
  this->need_sort_histogram_ =
      thrust::any_of(ctx->CUDACtx()->CTP(), feature_it, feature_it + n_features,
                     [=] XGBOOST_DEVICE(bst_feature_t fidx) {
                       if (!common::IsCat(feature_types, fidx)) {
                         return false;
                       }
                       auto n_cats = feature_segments[fidx + 1] - feature_segments[fidx];
                       return !common::UseOneHot(n_cats, max_cat_to_onehot);
                     });
}

// Sort the bins of partition-based features by the projected score, see the CPU
// implementation for details. The result is node-local bin indices.
common::Span<bst_bin_t const> GPUHistEvaluator::SortHistogram(
    Context const *ctx, common::Span<EvaluateSplitInputs const> d_inputs,
    EvaluateSplitSharedInputs const &shared_inputs) {
  std::size_t total_bins = shared_inputs.n_total_bins_per_tar;
  std::size_t n = d_inputs.size() * total_bins;
  this->sorted_idx_.resize(n);
  auto sorted_idx = dh::ToSpan(this->sorted_idx_);
  auto shared = shared_inputs;

  auto it = dh::make_counting_iterator<std::size_t>(0ul);
  if (this->feature_idx_.empty()) {
    // Cache the feature index of each bin.
    this->feature_idx_.resize(total_bins);
    thrust::transform(ctx->CUDACtx()->CTP(), it, it + total_bins, this->feature_idx_.begin(),
                      [=] XGBOOST_DEVICE(std::size_t i) -> bst_feature_t {
                        return dh::SegmentId(shared.feature_segments, i);
                      });
  }
  auto d_feature_idx = dh::ToSpan(this->feature_idx_);
  dh::TemporaryArray<std::size_t> partition_bins(n);
  auto partition_end = thrust::copy_if(
      ctx->CUDACtx()->CTP(), it, it + n, partition_bins.data(),
      [=] XGBOOST_DEVICE(std::size_t i) {
        return shared.IsPartition(d_feature_idx[i % total_bins]);
      });
  auto partition_n = partition_end - partition_bins.data();
  auto d_partition_bins = dh::ToSpan(partition_bins).subspan(0, partition_n);

  using SortPair = cuda::std::tuple<std::size_t, double>;
  dh::TemporaryArray<SortPair> keys(partition_n);
  auto d_keys = dh::ToSpan(keys);
  auto evaluator = this->GetEvaluator();
  thrust::for_each_n(ctx->CUDACtx()->CTP(), it, partition_n, [=] XGBOOST_DEVICE(std::size_t k) {
    auto global_idx = d_partition_bins[k];
    auto const &input = d_inputs[global_idx / total_bins];
    auto bin = global_idx % total_bins;
    double score = 0;
    auto n_targets = shared.Targets();
    for (bst_target_t t = 0; t < n_targets; ++t) {
      auto rounding = shared.roundings[t];
      auto child = rounding.ToFloatingPoint(input.histogram[t * total_bins + bin]);
      if (n_targets == 1) {
        score = evaluator.CalcWeightCat(child);
      } else {
        // Project the child weight onto the parent weight.
        auto parent = rounding.ToFloatingPoint(input.parent_sum[t]);
        score += evaluator.CalcWeightCat(parent) * evaluator.CalcWeightCat(child);
      }
    }
    d_keys[k] = cuda::std::make_tuple(global_idx, score);
  });
  // Sort an array segmented according to
  // - nodes
  // - features within each node
  // - scores within each feature
  thrust::stable_sort(ctx->CUDACtx()->CTP(), keys.data(), keys.data() + partition_n,
                      [=] XGBOOST_DEVICE(SortPair const &l, SortPair const &r) {
                        auto li = cuda::std::get<0>(l);
                        auto ri = cuda::std::get<0>(r);
                        auto l_node = li / total_bins;
                        auto r_node = ri / total_bins;
                        if (l_node != r_node) {
                          return l_node < r_node;
                        }
                        auto lfidx = d_feature_idx[li % total_bins];
                        auto rfidx = d_feature_idx[ri % total_bins];
                        if (lfidx != rfidx) {
                          return lfidx < rfidx;
                        }
                        return cuda::std::get<1>(l) < cuda::std::get<1>(r);
                      });
  thrust::for_each_n(ctx->CUDACtx()->CTP(), it, partition_n, [=] XGBOOST_DEVICE(std::size_t k) {
    sorted_idx[d_partition_bins[k]] = cuda::std::get<0>(d_keys[k]) % total_bins;
  });
  return sorted_idx;
}
}  // namespace xgboost::tree
