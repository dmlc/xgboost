/*!
 * Copyright 2020-2022 by XGBoost Contributors
 */
#include <algorithm>  // std::max
#include <limits>

#include "../../common/categorical.h"
#include "../../common/device_helpers.cuh"
#include "../../data/ellpack_page.cuh"
#include "evaluate_splits.cuh"
#include "expand_entry.cuh"

namespace xgboost {
namespace tree {

// With constraints
XGBOOST_DEVICE float LossChangeMissing(const GradientPairPrecise &scan,
                                       const GradientPairPrecise &missing,
                                       const GradientPairPrecise &parent_sum,
                                       const GPUTrainingParam &param, bst_node_t nidx,
                                       bst_feature_t fidx,
                                       TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
                                       bool &missing_left_out) {  // NOLINT
  float parent_gain = CalcGain(param, parent_sum);
  float missing_left_gain =
      evaluator.CalcSplitGain(param, nidx, fidx, GradStats(scan + missing),
                              GradStats(parent_sum - (scan + missing)));
  float missing_right_gain = evaluator.CalcSplitGain(
      param, nidx, fidx, GradStats(scan), GradStats(parent_sum - scan));

  if (missing_left_gain > missing_right_gain) {
    missing_left_out = true;
    return missing_left_gain - parent_gain;
  } else {
    missing_left_out = false;
    return missing_right_gain - parent_gain;
  }
}

/*!
 * \brief
 *
 * \tparam ReduceT     BlockReduce Type.
 * \tparam TempStorage Cub Shared memory
 *
 * \param begin
 * \param end
 * \param temp_storage Shared memory for intermediate result.
 */
template <int BLOCK_THREADS, typename ReduceT, typename TempStorageT,
          typename GradientSumT>
__device__ GradientSumT
ReduceFeature(common::Span<const GradientSumT> feature_histogram,
              TempStorageT* temp_storage) {
  __shared__ cub::Uninitialized<GradientSumT> uninitialized_sum;
  GradientSumT& shared_sum = uninitialized_sum.Alias();

  GradientSumT local_sum = GradientSumT();
  // For loop sums features into one block size
  auto begin = feature_histogram.data();
  auto end = begin + feature_histogram.size();
  for (auto itr = begin; itr < end; itr += BLOCK_THREADS) {
    bool thread_active = itr + threadIdx.x < end;
    // Scan histogram
    GradientSumT bin = thread_active ? *(itr + threadIdx.x) : GradientSumT();
    local_sum += bin;
  }
  local_sum = ReduceT(temp_storage->sum_reduce).Reduce(local_sum, cub::Sum());
  // Reduction result is stored in thread 0.
  if (threadIdx.x == 0) {
    shared_sum = local_sum;
  }
  cub::CTA_SYNC();
  return shared_sum;
}

/*! \brief Find the thread with best gain. */
template <int BLOCK_THREADS, typename ReduceT, typename ScanT, typename MaxReduceT,
          typename TempStorageT, typename GradientSumT, SplitType type>
__device__ void EvaluateFeature(
    int fidx, EvaluateSplitInputs<GradientSumT> inputs,
    TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
    common::Span<bst_feature_t> sorted_idx, size_t offset,
    DeviceSplitCandidate *best_split,  // shared memory storing best split
    TempStorageT *temp_storage         // temp memory for cub operations
) {
  // Use pointer from cut to indicate begin and end of bins for each feature.
  uint32_t gidx_begin = inputs.feature_segments[fidx];  // beginning bin
  uint32_t gidx_end =
      inputs.feature_segments[fidx + 1];  // end bin for i^th feature
  auto feature_hist = inputs.gradient_histogram.subspan(gidx_begin, gidx_end - gidx_begin);

  // Sum histogram bins for current feature
  GradientSumT const feature_sum =
      ReduceFeature<BLOCK_THREADS, ReduceT, TempStorageT, GradientSumT>(feature_hist, temp_storage);

  GradientPairPrecise const missing = inputs.parent_sum - GradientPairPrecise{feature_sum};
  float const null_gain = -std::numeric_limits<bst_float>::infinity();

  SumCallbackOp<GradientSumT> prefix_op = SumCallbackOp<GradientSumT>();
  for (int scan_begin = gidx_begin; scan_begin < gidx_end; scan_begin += BLOCK_THREADS) {
    bool thread_active = (scan_begin + threadIdx.x) < gidx_end;

    auto calc_bin_value = [&]() {
      GradientSumT bin;
      switch (type) {
        case kOneHot: {
          auto rest =
              thread_active ? inputs.gradient_histogram[scan_begin + threadIdx.x] : GradientSumT();
          bin = GradientSumT{inputs.parent_sum - GradientPairPrecise{rest} - missing};  // NOLINT
          break;
        }
        case kNum: {
          bin =
              thread_active ? inputs.gradient_histogram[scan_begin + threadIdx.x] : GradientSumT();
          ScanT(temp_storage->scan).ExclusiveScan(bin, bin, cub::Sum(), prefix_op);
          break;
        }
        case kPart: {
          auto rest = thread_active
                          ? inputs.gradient_histogram[sorted_idx[scan_begin + threadIdx.x] - offset]
                          : GradientSumT();
          // No min value for cat feature, use inclusive scan.
          ScanT(temp_storage->scan).InclusiveScan(rest, rest, cub::Sum(), prefix_op);
          bin = GradientSumT{inputs.parent_sum - GradientPairPrecise{rest} - missing};  // NOLINT
          break;
        }
      }
      return bin;
    };
    auto bin = calc_bin_value();
    // Whether the gradient of missing values is put to the left side.
    bool missing_left = true;
    float gain = null_gain;
    if (thread_active) {
      gain = LossChangeMissing(GradientPairPrecise{bin}, missing, inputs.parent_sum, inputs.param,
                               inputs.nidx, fidx, evaluator, missing_left);
    }

    __syncthreads();

    // Find thread with best gain
    cub::KeyValuePair<int, float> tuple(threadIdx.x, gain);
    cub::KeyValuePair<int, float> best =
        MaxReduceT(temp_storage->max_reduce).Reduce(tuple, cub::ArgMax());

    __shared__ cub::KeyValuePair<int, float> block_max;
    if (threadIdx.x == 0) {
      block_max = best;
    }

    cub::CTA_SYNC();

    // Best thread updates the split
    if (threadIdx.x == block_max.key) {
      switch (type) {
        case kNum: {
          // Use pointer from cut to indicate begin and end of bins for each feature.
          uint32_t gidx_begin = inputs.feature_segments[fidx];  // beginning bin
          int split_gidx = (scan_begin + threadIdx.x) - 1;
          float fvalue;
          if (split_gidx < static_cast<int>(gidx_begin)) {
            fvalue = inputs.min_fvalue[fidx];
          } else {
            fvalue = inputs.feature_values[split_gidx];
          }
          GradientPairPrecise left =
              missing_left ? GradientPairPrecise{bin} + missing : GradientPairPrecise{bin};
          GradientPairPrecise right = inputs.parent_sum - left;
          best_split->Update(gain, missing_left ? kLeftDir : kRightDir, fvalue, fidx, left, right,
                             false, inputs.param);
          break;
        }
        case kOneHot: {
          int32_t split_gidx = (scan_begin + threadIdx.x);
          float fvalue = inputs.feature_values[split_gidx];
          GradientPairPrecise left =
              missing_left ? GradientPairPrecise{bin} + missing : GradientPairPrecise{bin};
          GradientPairPrecise right = inputs.parent_sum - left;
          best_split->Update(gain, missing_left ? kLeftDir : kRightDir, fvalue, fidx, left, right,
                             true, inputs.param);
          break;
        }
        case kPart: {
          int32_t split_gidx = (scan_begin + threadIdx.x);
          float fvalue = inputs.feature_values[split_gidx];
          GradientPairPrecise left =
              missing_left ? GradientPairPrecise{bin} + missing : GradientPairPrecise{bin};
          GradientPairPrecise right = inputs.parent_sum - left;
          auto best_thresh = block_max.key;  // index of best threshold inside a feature.
          best_split->Update(gain, missing_left ? kLeftDir : kRightDir, best_thresh, fidx, left,
                             right, true, inputs.param);
          break;
        }
      }
    }
    cub::CTA_SYNC();
  }
}

template <int BLOCK_THREADS, typename GradientSumT>
__global__ void EvaluateSplitsKernel(EvaluateSplitInputs<GradientSumT> left,
                                     EvaluateSplitInputs<GradientSumT> right,
                                     common::Span<bst_feature_t> sorted_idx,
                                     TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
                                     common::Span<DeviceSplitCandidate> out_candidates) {
  // KeyValuePair here used as threadIdx.x -> gain_value
  using ArgMaxT = cub::KeyValuePair<int, float>;
  using BlockScanT =
      cub::BlockScan<GradientSumT, BLOCK_THREADS, cub::BLOCK_SCAN_WARP_SCANS>;
  using MaxReduceT = cub::BlockReduce<ArgMaxT, BLOCK_THREADS>;

  using SumReduceT = cub::BlockReduce<GradientSumT, BLOCK_THREADS>;

  union TempStorage {
    typename BlockScanT::TempStorage scan;
    typename MaxReduceT::TempStorage max_reduce;
    typename SumReduceT::TempStorage sum_reduce;
  };

  // Aligned && shared storage for best_split
  __shared__ cub::Uninitialized<DeviceSplitCandidate> uninitialized_split;
  DeviceSplitCandidate& best_split = uninitialized_split.Alias();
  __shared__ TempStorage temp_storage;

  if (threadIdx.x == 0) {
    best_split = DeviceSplitCandidate();
  }

  __syncthreads();

  // If this block is working on the left or right node
  bool is_left = blockIdx.x < left.feature_set.size();
  EvaluateSplitInputs<GradientSumT>& inputs = is_left ? left : right;

  // One block for each feature. Features are sampled, so fidx != blockIdx.x
  int fidx = inputs.feature_set[is_left ? blockIdx.x
                                        : blockIdx.x - left.feature_set.size()];

  if (common::IsCat(inputs.feature_types, fidx)) {
    auto n_bins_in_feat = inputs.feature_segments[fidx + 1] - inputs.feature_segments[fidx];
    if (common::UseOneHot(n_bins_in_feat, inputs.param.max_cat_to_onehot)) {
      EvaluateFeature<BLOCK_THREADS, SumReduceT, BlockScanT, MaxReduceT, TempStorage, GradientSumT,
                      kOneHot>(fidx, inputs, evaluator, sorted_idx, 0, &best_split, &temp_storage);
    } else {
      auto node_sorted_idx = is_left ? sorted_idx.first(inputs.feature_values.size())
                                     : sorted_idx.last(inputs.feature_values.size());
      size_t offset = is_left ? 0 : inputs.feature_values.size();
      EvaluateFeature<BLOCK_THREADS, SumReduceT, BlockScanT, MaxReduceT, TempStorage, GradientSumT,
                      kPart>(fidx, inputs, evaluator, node_sorted_idx, offset, &best_split,
                             &temp_storage);
    }
  } else {
    EvaluateFeature<BLOCK_THREADS, SumReduceT, BlockScanT, MaxReduceT, TempStorage, GradientSumT,
                    kNum>(fidx, inputs, evaluator, sorted_idx, 0, &best_split, &temp_storage);
  }

  cub::CTA_SYNC();
  if (threadIdx.x == 0) {
    // Record best loss for each feature
    out_candidates[blockIdx.x] = best_split;
  }
}

__device__ DeviceSplitCandidate operator+(const DeviceSplitCandidate& a,
                                          const DeviceSplitCandidate& b) {
  return b.loss_chg > a.loss_chg ? b : a;
}

/**
 * \brief Set the bits for categorical splits based on the split threshold.
 */
template <typename GradientSumT>
__device__ void SortBasedSplit(EvaluateSplitInputs<GradientSumT> const &input,
                               common::Span<bst_feature_t const> d_sorted_idx, bst_feature_t fidx,
                               bool is_left, common::Span<common::CatBitField::value_type> out,
                               DeviceSplitCandidate *p_out_split) {
  auto &out_split = *p_out_split;
  out_split.split_cats = common::CatBitField{out};
  auto node_sorted_idx =
      is_left ? d_sorted_idx.subspan(0, input.feature_values.size())
              : d_sorted_idx.subspan(input.feature_values.size(), input.feature_values.size());
  size_t node_offset = is_left ? 0 : input.feature_values.size();
  auto best_thresh = out_split.PopBestThresh();
  auto f_sorted_idx =
      node_sorted_idx.subspan(input.feature_segments[fidx], input.FeatureBins(fidx));
  if (out_split.dir != kLeftDir) {
    // forward, missing on right
    auto beg = dh::tcbegin(f_sorted_idx);
    // Don't put all the categories into one side
    auto boundary = std::min(static_cast<size_t>((best_thresh + 1)), (f_sorted_idx.size() - 1));
    boundary = std::max(boundary, static_cast<size_t>(1ul));
    auto end = beg + boundary;
    thrust::for_each(thrust::seq, beg, end, [&](auto c) {
      auto cat = input.feature_values[c - node_offset];
      assert(!out_split.split_cats.Check(cat) && "already set");
      out_split.SetCat(cat);
    });
  } else {
    assert((f_sorted_idx.size() - best_thresh + 1) != 0 && " == 0");
    thrust::for_each(thrust::seq, dh::tcrbegin(f_sorted_idx),
                     dh::tcrbegin(f_sorted_idx) + (f_sorted_idx.size() - best_thresh), [&](auto c) {
                       auto cat = input.feature_values[c - node_offset];
                       out_split.SetCat(cat);
                     });
  }
}

template <typename GradientSumT>
void GPUHistEvaluator<GradientSumT>::EvaluateSplits(
    EvaluateSplitInputs<GradientSumT> left, EvaluateSplitInputs<GradientSumT> right,
    TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
    common::Span<DeviceSplitCandidate> out_splits) {
  if (!split_cats_.empty()) {
    this->SortHistogram(left, right, evaluator);
  }

  size_t combined_num_features = left.feature_set.size() + right.feature_set.size();
  dh::TemporaryArray<DeviceSplitCandidate> feature_best_splits(combined_num_features);

  // One block for each feature
  uint32_t constexpr kBlockThreads = 256;
  dh::LaunchKernel {static_cast<uint32_t>(combined_num_features), kBlockThreads, 0}(
      EvaluateSplitsKernel<kBlockThreads, GradientSumT>, left, right, this->SortedIdx(left),
      evaluator, dh::ToSpan(feature_best_splits));

  // Reduce to get best candidate for left and right child over all features
  auto reduce_offset = dh::MakeTransformIterator<size_t>(thrust::make_counting_iterator(0llu),
                                                         [=] __device__(size_t idx) -> size_t {
                                                           if (idx == 0) {
                                                             return 0;
                                                           }
                                                           if (idx == 1) {
                                                             return left.feature_set.size();
                                                           }
                                                           if (idx == 2) {
                                                             return combined_num_features;
                                                           }
                                                           return 0;
                                                         });
  size_t temp_storage_bytes = 0;
  auto num_segments = out_splits.size();
  cub::DeviceSegmentedReduce::Sum(nullptr, temp_storage_bytes, feature_best_splits.data(),
                                  out_splits.data(), num_segments, reduce_offset,
                                  reduce_offset + 1);
  dh::TemporaryArray<int8_t> temp(temp_storage_bytes);
  cub::DeviceSegmentedReduce::Sum(temp.data().get(), temp_storage_bytes, feature_best_splits.data(),
                                  out_splits.data(), num_segments, reduce_offset,
                                  reduce_offset + 1);
}

template <typename GradientSumT>
void GPUHistEvaluator<GradientSumT>::CopyToHost(EvaluateSplitInputs<GradientSumT> const &input,
                                                common::Span<CatST> cats_out) {
  if (has_sort_) {
    dh::CUDAEvent event;
    event.Record(dh::DefaultStream());
    auto h_cats = this->HostCatStorage(input.nidx);
    copy_stream_.View().Wait(event);
    dh::safe_cuda(cudaMemcpyAsync(h_cats.data(), cats_out.data(), cats_out.size_bytes(),
                                  cudaMemcpyDeviceToHost, copy_stream_.View()));
  }
}

template <typename GradientSumT>
void GPUHistEvaluator<GradientSumT>::EvaluateSplits(GPUExpandEntry candidate,
                                                    EvaluateSplitInputs<GradientSumT> left,
                                                    EvaluateSplitInputs<GradientSumT> right,
                                                    common::Span<GPUExpandEntry> out_entries) {
  auto evaluator = this->tree_evaluator_.template GetEvaluator<GPUTrainingParam>();

  dh::TemporaryArray<DeviceSplitCandidate> splits_out_storage(2);
  auto out_splits = dh::ToSpan(splits_out_storage);
  this->EvaluateSplits(left, right, evaluator, out_splits);

  auto d_sorted_idx = this->SortedIdx(left);
  auto d_entries = out_entries;
  auto cats_out = this->DeviceCatStorage(left.nidx);
  // turn candidate into entry, along with hanlding sort based split.
  dh::LaunchN(right.feature_set.empty() ? 1 : 2, [=] __device__(size_t i) {
    auto const &input = i == 0 ? left : right;
    auto &split = out_splits[i];
    auto fidx = out_splits[i].findex;

    if (split.is_cat &&
        !common::UseOneHot(input.FeatureBins(fidx), input.param.max_cat_to_onehot)) {
      bool is_left = i == 0;
      auto out = is_left ? cats_out.first(cats_out.size() / 2) : cats_out.last(cats_out.size() / 2);
      SortBasedSplit(input, d_sorted_idx, fidx, is_left, out, &out_splits[i]);
    }

    float base_weight =
        evaluator.CalcWeight(input.nidx, input.param, GradStats{split.left_sum + split.right_sum});
    float left_weight = evaluator.CalcWeight(input.nidx, input.param, GradStats{split.left_sum});
    float right_weight = evaluator.CalcWeight(input.nidx, input.param, GradStats{split.right_sum});

    d_entries[i] = GPUExpandEntry{input.nidx,  candidate.depth + 1, out_splits[i],
                                  base_weight, left_weight,         right_weight};
  });

  this->CopyToHost(left, cats_out);
}

template <typename GradientSumT>
GPUExpandEntry GPUHistEvaluator<GradientSumT>::EvaluateSingleSplit(
    EvaluateSplitInputs<GradientSumT> input, float weight) {
  dh::TemporaryArray<DeviceSplitCandidate> splits_out(1);
  auto out_split = dh::ToSpan(splits_out);
  auto evaluator = tree_evaluator_.GetEvaluator<GPUTrainingParam>();
  this->EvaluateSplits(input, {}, evaluator, out_split);

  auto cats_out = this->DeviceCatStorage(input.nidx);
  auto d_sorted_idx = this->SortedIdx(input);

  dh::TemporaryArray<GPUExpandEntry> entries(1);
  auto d_entries = entries.data().get();
  dh::LaunchN(1, [=] __device__(size_t i) {
    auto &split = out_split[i];
    auto fidx = out_split[i].findex;

    if (split.is_cat &&
        !common::UseOneHot(input.FeatureBins(fidx), input.param.max_cat_to_onehot)) {
      SortBasedSplit(input, d_sorted_idx, fidx, true, cats_out, &out_split[i]);
    }

    float left_weight = evaluator.CalcWeight(0, input.param, GradStats{split.left_sum});
    float right_weight = evaluator.CalcWeight(0, input.param, GradStats{split.right_sum});
    d_entries[0] = GPUExpandEntry(0, 0, split, weight, left_weight, right_weight);
  });
  this->CopyToHost(input, cats_out);

  GPUExpandEntry root_entry;
  dh::safe_cuda(cudaMemcpyAsync(&root_entry, entries.data().get(),
                                sizeof(GPUExpandEntry) * entries.size(), cudaMemcpyDeviceToHost));
  return root_entry;
}

template class GPUHistEvaluator<GradientPair>;
template class GPUHistEvaluator<GradientPairPrecise>;
}  // namespace tree
}  // namespace xgboost
