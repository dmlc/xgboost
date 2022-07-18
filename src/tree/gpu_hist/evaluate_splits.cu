/*!
 * Copyright 2020-2022 by XGBoost Contributors
 */
#include <algorithm>  // std::max
#include <vector>
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
  const auto left_sum = scan + missing;
  float missing_left_gain =
      evaluator.CalcSplitGain(param, nidx, fidx, left_sum, parent_sum - left_sum);
  float missing_right_gain = evaluator.CalcSplitGain(param, nidx, fidx, scan, parent_sum - scan);

  missing_left_out = missing_left_gain > missing_right_gain;
  return missing_left_out?missing_left_gain:missing_right_gain;
}


template <int BLOCK_THREADS, typename GradientSumT>
class EvaluateSplitAgent {
 public:
  using ArgMaxT = cub::KeyValuePair<int, float>;
  using BlockScanT = cub::BlockScan<GradientSumT, BLOCK_THREADS>;
  using MaxReduceT =
      cub::WarpReduce<ArgMaxT>;
  using SumReduceT = cub::WarpReduce<GradientSumT>;
  struct TempStorage {
    typename BlockScanT::TempStorage scan;
    typename MaxReduceT::TempStorage max_reduce;
    typename SumReduceT::TempStorage sum_reduce;
  };

  const int fidx;
  const int nidx;
  const float min_fvalue;
  const uint32_t gidx_begin;  // beginning bin
  const uint32_t gidx_end;    // end bin for i^th feature
  const dh::LDGIterator<float> feature_values;
  const GradientSumT *node_histogram;
  const GradientSumT parent_sum;
  const GradientSumT missing;
  const GPUTrainingParam &param;
  const TreeEvaluator::SplitEvaluator<GPUTrainingParam> &evaluator;
  TempStorage *temp_storage;
  SumCallbackOp<GradientSumT> prefix_op;
  static float constexpr kNullGain = -std::numeric_limits<bst_float>::infinity();

  __device__ EvaluateSplitAgent(TempStorage *temp_storage, int fidx,
                                const EvaluateSplitInputs &inputs,
                                const EvaluateSplitSharedInputs &shared_inputs,
                                const TreeEvaluator::SplitEvaluator<GPUTrainingParam> &evaluator)
      : temp_storage(temp_storage),
        nidx(inputs.nidx),
        fidx(fidx),
        min_fvalue(__ldg(shared_inputs.min_fvalue.data() + fidx)),
        gidx_begin(__ldg(shared_inputs.feature_segments.data() + fidx)),
        gidx_end(__ldg(shared_inputs.feature_segments.data() + fidx + 1)),
        feature_values(shared_inputs.feature_values.data()),
        node_histogram(inputs.gradient_histogram.data()),
        parent_sum(dh::LDGIterator<GradientSumT>(&inputs.parent_sum)[0]),
        param(shared_inputs.param),
        evaluator(evaluator),
        missing(parent_sum - ReduceFeature()) {}
  __device__ GradientSumT ReduceFeature() {
    GradientSumT local_sum;
    for (int idx = gidx_begin + threadIdx.x; idx < gidx_end; idx += BLOCK_THREADS) {
      local_sum += LoadGpair(node_histogram + idx);
    }
    local_sum = SumReduceT(temp_storage->sum_reduce).Sum(local_sum);
    // Broadcast result from thread 0
    return {__shfl_sync(0xffffffff, local_sum.GetGrad(), 0),
            __shfl_sync(0xffffffff, local_sum.GetHess(), 0)};
  }

  // Load using efficient 128 vector load instruction
  __device__ __forceinline__ GradientSumT LoadGpair(const GradientSumT *ptr) {
    static_assert(sizeof(GradientSumT) == sizeof(float4),
                  "Vector type size does not match gradient pair size.");
    float4 tmp = *reinterpret_cast<const float4 *>(ptr);
    return *reinterpret_cast<const GradientSumT *>(&tmp);
  }

  __device__ __forceinline__ void Numerical(DeviceSplitCandidate *__restrict__ best_split) {
    for (int scan_begin = gidx_begin; scan_begin < gidx_end; scan_begin += BLOCK_THREADS) {
      bool thread_active = (scan_begin + threadIdx.x) < gidx_end;
      GradientSumT bin =
          thread_active ? LoadGpair(node_histogram + scan_begin + threadIdx.x) : GradientSumT();
      BlockScanT(temp_storage->scan).ExclusiveScan(bin, bin, cub::Sum(), prefix_op);
      // Whether the gradient of missing values is put to the left side.
      bool missing_left = true;
      float gain = thread_active ? LossChangeMissing(bin, missing, parent_sum, param, nidx, fidx,
                                                     evaluator, missing_left)
                                 : kNullGain;

      // Find thread with best gain
      auto best = MaxReduceT(temp_storage->max_reduce).Reduce({threadIdx.x, gain}, cub::ArgMax());
      // This reduce result is only valid in thread 0
      // broadcast to the rest of the warp
      auto best_thread = __shfl_sync(0xffffffff, best.key, 0);

      // Best thread updates the split
      if (threadIdx.x == best_thread) {
        // Use pointer from cut to indicate begin and end of bins for each feature.
        int split_gidx = (scan_begin + threadIdx.x) - 1;
        float fvalue =
            split_gidx < static_cast<int>(gidx_begin) ? min_fvalue : feature_values[split_gidx];
        GradientPairPrecise left = missing_left ? bin + missing : bin;
        GradientPairPrecise right = parent_sum - left;
        best_split->Update(gain, missing_left ? kLeftDir : kRightDir, fvalue, fidx, left, right,
                           false, param);
      }
    }
  }

  __device__ __forceinline__ void OneHot(DeviceSplitCandidate *__restrict__ best_split) {
    for (int scan_begin = gidx_begin; scan_begin < gidx_end; scan_begin += BLOCK_THREADS) {
      bool thread_active = (scan_begin + threadIdx.x) < gidx_end;

      auto rest =
          thread_active ? LoadGpair(node_histogram + scan_begin + threadIdx.x) : GradientSumT();
      GradientSumT bin = parent_sum - rest - missing;
      // Whether the gradient of missing values is put to the left side.
      bool missing_left = true;
      float gain = thread_active ? LossChangeMissing(bin, missing, parent_sum, param, nidx, fidx,
                                                     evaluator, missing_left)
                                 : kNullGain;

      // Find thread with best gain
      auto best = MaxReduceT(temp_storage->max_reduce).Reduce({threadIdx.x, gain}, cub::ArgMax());
      // This reduce result is only valid in thread 0
      // broadcast to the rest of the warp
      auto best_thread = __shfl_sync(0xffffffff, best.key, 0);
      // Best thread updates the split
      if (threadIdx.x == best_thread) {
        int32_t split_gidx = (scan_begin + threadIdx.x);
        float fvalue = feature_values[split_gidx];
        GradientPairPrecise left =
            missing_left ? bin + missing : bin;
        GradientPairPrecise right = parent_sum - left;
        best_split->Update(gain, missing_left ? kLeftDir : kRightDir, fvalue, fidx, left, right,
                           true, param);
      }
    }
  }
  __device__ __forceinline__ void Partition(DeviceSplitCandidate *__restrict__ best_split,
                                         bst_feature_t * __restrict__ sorted_idx,
                                         std::size_t offset) {
    for (int scan_begin = gidx_begin; scan_begin < gidx_end; scan_begin += BLOCK_THREADS) {
      bool thread_active = (scan_begin + threadIdx.x) < gidx_end;

      auto rest = thread_active
                      ? LoadGpair(node_histogram + sorted_idx[scan_begin + threadIdx.x] - offset)
                      : GradientSumT();
      // No min value for cat feature, use inclusive scan.
      BlockScanT(temp_storage->scan).InclusiveSum(rest, rest,  prefix_op);
      GradientSumT bin = parent_sum - rest - missing;

      // Whether the gradient of missing values is put to the left side.
      bool missing_left = true;
      float gain = thread_active ? LossChangeMissing(bin, missing, parent_sum, param, nidx, fidx,
                                                     evaluator, missing_left)
                                 : kNullGain;


      // Find thread with best gain
      auto best =
          MaxReduceT(temp_storage->max_reduce).Reduce({threadIdx.x, gain}, cub::ArgMax());
      // This reduce result is only valid in thread 0
      // broadcast to the rest of the warp
      auto best_thread = __shfl_sync(0xffffffff, best.key, 0);
      // Best thread updates the split
      if (threadIdx.x == best_thread) {
        GradientPairPrecise left = missing_left ? bin + missing : bin;
        GradientPairPrecise right = parent_sum - left;
        auto best_thresh =
            threadIdx.x + (scan_begin - gidx_begin);  // index of best threshold inside a feature.
        best_split->Update(gain, missing_left ? kLeftDir : kRightDir, best_thresh, fidx, left,
                           right, true, param);
      }
    }
  }
};

template <int BLOCK_THREADS, typename GradientSumT>
__global__ __launch_bounds__(BLOCK_THREADS) void EvaluateSplitsKernel(
    bst_feature_t number_active_features, common::Span<const EvaluateSplitInputs> d_inputs,
    const EvaluateSplitSharedInputs shared_inputs, common::Span<bst_feature_t> sorted_idx,
    const TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
    common::Span<DeviceSplitCandidate> out_candidates) {
  // Aligned && shared storage for best_split
  __shared__ cub::Uninitialized<DeviceSplitCandidate> uninitialized_split;
  DeviceSplitCandidate &best_split = uninitialized_split.Alias();

  if (threadIdx.x == 0) {
    best_split = DeviceSplitCandidate();
  }

  __syncthreads();

  // Allocate blocks to one feature of one node
  const auto input_idx = blockIdx.x / number_active_features;
  const EvaluateSplitInputs &inputs = d_inputs[input_idx];
  // One block for each feature. Features are sampled, so fidx != blockIdx.x

  int fidx = inputs.feature_set[blockIdx.x % number_active_features];

  using AgentT = EvaluateSplitAgent<BLOCK_THREADS, GradientSumT>;
  __shared__ typename AgentT::TempStorage temp_storage;
  AgentT agent(&temp_storage, fidx, inputs, shared_inputs, evaluator);

  if (common::IsCat(shared_inputs.feature_types, fidx)) {
    auto n_bins_in_feat =
        shared_inputs.feature_segments[fidx + 1] - shared_inputs.feature_segments[fidx];
    if (common::UseOneHot(n_bins_in_feat, shared_inputs.param.max_cat_to_onehot)) {
      agent.OneHot(&best_split);
    } else {
      auto total_bins = shared_inputs.feature_values.size();
      size_t offset = total_bins * input_idx;
      auto node_sorted_idx = sorted_idx.subspan(offset, total_bins);
      agent.Partition(&best_split, node_sorted_idx.data(), offset);
    }
  } else {
    agent.Numerical(&best_split);
  }

  cub::CTA_SYNC();
  if (threadIdx.x == 0) {
    // Record best loss for each feature
    out_candidates[blockIdx.x] = best_split;
  }
}

__device__ DeviceSplitCandidate operator+(const DeviceSplitCandidate &a,
                                          const DeviceSplitCandidate &b) {
  return b.loss_chg > a.loss_chg ? b : a;
}

/**
 * \brief Set the bits for categorical splits based on the split threshold.
 */
__device__ void SetCategoricalSplit(const EvaluateSplitSharedInputs &shared_inputs,
                                    common::Span<bst_feature_t const> d_sorted_idx,
                                    bst_feature_t fidx, std::size_t input_idx,
                                    common::Span<common::CatBitField::value_type> out,
                                    DeviceSplitCandidate *p_out_split) {
  auto &out_split = *p_out_split;
  out_split.split_cats = common::CatBitField{out};

  // Simple case for one hot split
  if (common::UseOneHot(shared_inputs.FeatureBins(fidx), shared_inputs.param.max_cat_to_onehot)) {
    out_split.split_cats.Set(common::AsCat(out_split.fvalue));
    return;
  }

  auto node_sorted_idx = d_sorted_idx.subspan(shared_inputs.feature_values.size() * input_idx,
                                              shared_inputs.feature_values.size());
  size_t node_offset = input_idx * shared_inputs.feature_values.size();
  auto best_thresh = out_split.PopBestThresh();
  auto f_sorted_idx = node_sorted_idx.subspan(shared_inputs.feature_segments[fidx],
                                              shared_inputs.FeatureBins(fidx));
  if (out_split.dir != kLeftDir) {
    // forward, missing on right
    auto beg = dh::tcbegin(f_sorted_idx);
    // Don't put all the categories into one side
    auto boundary = std::min(static_cast<size_t>((best_thresh + 1)), (f_sorted_idx.size() - 1));
    boundary = std::max(boundary, static_cast<size_t>(1ul));
    auto end = beg + boundary;
    thrust::for_each(thrust::seq, beg, end, [&](auto c) {
      auto cat = shared_inputs.feature_values[c - node_offset];
      assert(!out_split.split_cats.Check(cat) && "already set");
      out_split.SetCat(cat);
    });
  } else {
    assert((f_sorted_idx.size() - best_thresh + 1) != 0 && " == 0");
    thrust::for_each(thrust::seq, dh::tcrbegin(f_sorted_idx),
                     dh::tcrbegin(f_sorted_idx) + (f_sorted_idx.size() - best_thresh), [&](auto c) {
                       auto cat = shared_inputs.feature_values[c - node_offset];
                       out_split.SetCat(cat);
                     });
  }
}

template <typename GradientSumT>
void GPUHistEvaluator<GradientSumT>::LaunchEvaluateSplits(
    bst_feature_t number_active_features, common::Span<const EvaluateSplitInputs> d_inputs,
    EvaluateSplitSharedInputs shared_inputs,
    TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
    common::Span<DeviceSplitCandidate> out_splits) {
  if (need_sort_histogram_) {
    this->SortHistogram(d_inputs, shared_inputs, evaluator);
  }

  size_t combined_num_features = number_active_features * d_inputs.size();
  dh::TemporaryArray<DeviceSplitCandidate> feature_best_splits(combined_num_features);

  // One block for each feature
  uint32_t constexpr kBlockThreads = 32;
  dh::LaunchKernel {static_cast<uint32_t>(combined_num_features), kBlockThreads, 0}(
      EvaluateSplitsKernel<kBlockThreads, GradientSumT>, number_active_features, d_inputs,
      shared_inputs, this->SortedIdx(d_inputs.size(), shared_inputs.feature_values.size()),
      evaluator, dh::ToSpan(feature_best_splits));

  // Reduce to get best candidate for left and right child over all features
  auto reduce_offset = dh::MakeTransformIterator<size_t>(
      thrust::make_counting_iterator(0llu),
      [=] __device__(size_t idx) -> size_t { return idx * number_active_features; });
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
void GPUHistEvaluator<GradientSumT>::CopyToHost(const std::vector<bst_node_t> &nidx) {
  if (!has_categoricals_) return;
  auto d_cats = this->DeviceCatStorage(nidx);
  auto h_cats = this->HostCatStorage(nidx);
  dh::CUDAEvent event;
  event.Record(dh::DefaultStream());
  for (auto idx : nidx) {
    copy_stream_.View().Wait(event);
    dh::safe_cuda(cudaMemcpyAsync(
        h_cats.GetNodeCatStorage(idx).data(), d_cats.GetNodeCatStorage(idx).data(),
        d_cats.GetNodeCatStorage(idx).size_bytes(), cudaMemcpyDeviceToHost, copy_stream_.View()));
  }
}

template <typename GradientSumT>
void GPUHistEvaluator<GradientSumT>::EvaluateSplits(
    const std::vector<bst_node_t> &nidx, bst_feature_t number_active_features,
    common::Span<const EvaluateSplitInputs> d_inputs, EvaluateSplitSharedInputs shared_inputs,
    common::Span<GPUExpandEntry> out_entries) {
  auto evaluator = this->tree_evaluator_.template GetEvaluator<GPUTrainingParam>();

  dh::TemporaryArray<DeviceSplitCandidate> splits_out_storage(d_inputs.size());
  auto out_splits = dh::ToSpan(splits_out_storage);
  this->LaunchEvaluateSplits(number_active_features, d_inputs, shared_inputs, evaluator,
                             out_splits);

  auto d_sorted_idx = this->SortedIdx(d_inputs.size(), shared_inputs.feature_values.size());
  auto d_entries = out_entries;
  auto device_cats_accessor = this->DeviceCatStorage(nidx);
  // turn candidate into entry, along with handling sort based split.
  dh::LaunchN(d_inputs.size(), [=] __device__(size_t i) mutable {
    auto const input = d_inputs[i];
    auto &split = out_splits[i];
    // Subtract parent gain here
    // As it is constant, this is more efficient than doing it during every split evaluation
    float parent_gain = CalcGain(shared_inputs.param, input.parent_sum);
    split.loss_chg -= parent_gain;
    auto fidx = out_splits[i].findex;

    if (split.is_cat) {
      SetCategoricalSplit(shared_inputs, d_sorted_idx, fidx, i,
                          device_cats_accessor.GetNodeCatStorage(input.nidx), &out_splits[i]);
    }

    float base_weight = evaluator.CalcWeight(input.nidx, shared_inputs.param,
                                             GradStats{split.left_sum + split.right_sum});
    float left_weight =
        evaluator.CalcWeight(input.nidx, shared_inputs.param, GradStats{split.left_sum});
    float right_weight =
        evaluator.CalcWeight(input.nidx, shared_inputs.param, GradStats{split.right_sum});

    d_entries[i] = GPUExpandEntry{input.nidx,  input.depth, out_splits[i],
                                  base_weight, left_weight, right_weight};
  });

  this->CopyToHost(nidx);
}

template <typename GradientSumT>
GPUExpandEntry GPUHistEvaluator<GradientSumT>::EvaluateSingleSplit(
    EvaluateSplitInputs input, EvaluateSplitSharedInputs shared_inputs) {
  dh::device_vector<EvaluateSplitInputs> inputs = std::vector<EvaluateSplitInputs>{input};
  dh::TemporaryArray<GPUExpandEntry> out_entries(1);
  this->EvaluateSplits({input.nidx}, input.feature_set.size(), dh::ToSpan(inputs), shared_inputs,
                       dh::ToSpan(out_entries));
  GPUExpandEntry root_entry;
  dh::safe_cuda(cudaMemcpyAsync(&root_entry, out_entries.data().get(), sizeof(GPUExpandEntry),
                                cudaMemcpyDeviceToHost));
  return root_entry;
}

template class GPUHistEvaluator<GradientPairPrecise>;
}  // namespace tree
}  // namespace xgboost
