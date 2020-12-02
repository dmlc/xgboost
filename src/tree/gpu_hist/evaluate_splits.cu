/*!
 * Copyright 2020 by XGBoost Contributors
 */
#include <limits>
#include "evaluate_splits.cuh"
#include "../../common/categorical.h"

namespace xgboost {
namespace tree {

// With constraints
template <typename GradientPairT>
XGBOOST_DEVICE float
LossChangeMissing(const GradientPairT &scan, const GradientPairT &missing,
                  const GradientPairT &parent_sum,
                  const GPUTrainingParam &param,
                  bst_node_t nidx,
                  bst_feature_t fidx,
                  TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
                  bool &missing_left_out) { // NOLINT
  float parent_gain = CalcGain(param, parent_sum);
  float missing_left_gain =
      evaluator.CalcSplitGain(param, nidx, fidx, GradStats(scan + missing),
                              GradStats(parent_sum - (scan + missing)));
  float missing_right_gain = evaluator.CalcSplitGain(
      param, nidx, fidx, GradStats(scan), GradStats(parent_sum - scan));

  if (missing_left_gain >= missing_right_gain) {
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

template <typename GradientSumT, typename TempStorageT> struct OneHotBin {
  GradientSumT __device__ operator()(
      bool thread_active, uint32_t scan_begin,
      SumCallbackOp<GradientSumT>*,
      GradientSumT const &missing,
      EvaluateSplitInputs<GradientSumT> const &inputs, TempStorageT *) {
    GradientSumT bin = thread_active
                           ? inputs.gradient_histogram[scan_begin + threadIdx.x]
                           : GradientSumT();
    auto rest = inputs.parent_sum - bin - missing;
    return rest;
  }
};

template <typename GradientSumT>
struct UpdateOneHot {
  void __device__ operator()(bool missing_left, uint32_t scan_begin, float gain,
                             bst_feature_t fidx, GradientSumT const &missing,
                             GradientSumT const &bin,
                             EvaluateSplitInputs<GradientSumT> const &inputs,
                             DeviceSplitCandidate *best_split) {
    int split_gidx = (scan_begin + threadIdx.x);
    float fvalue = inputs.feature_values[split_gidx];
    GradientSumT left = missing_left ? bin + missing : bin;
    GradientSumT right = inputs.parent_sum - left;
    best_split->Update(gain, missing_left ? kLeftDir : kRightDir, fvalue, fidx,
                       GradientPair(left), GradientPair(right), true,
                       inputs.param);
  }
};

template <typename GradientSumT, typename TempStorageT, typename ScanT>
struct NumericBin {
  GradientSumT __device__ operator()(bool thread_active, uint32_t scan_begin,
                                     SumCallbackOp<GradientSumT>* prefix_callback,
                                     GradientSumT const &missing,
                                     EvaluateSplitInputs<GradientSumT> inputs,
                                     TempStorageT *temp_storage) {
    GradientSumT bin = thread_active
                       ? inputs.gradient_histogram[scan_begin + threadIdx.x]
                       : GradientSumT();
    ScanT(temp_storage->scan).ExclusiveScan(bin, bin, cub::Sum(), *prefix_callback);
    return bin;
  }
};

template <typename GradientSumT>
struct UpdateNumeric {
  void __device__ operator()(bool missing_left, uint32_t scan_begin, float gain,
                             bst_feature_t fidx, GradientSumT const &missing,
                             GradientSumT const &bin,
                             EvaluateSplitInputs<GradientSumT> const &inputs,
                             DeviceSplitCandidate *best_split) {
    // Use pointer from cut to indicate begin and end of bins for each feature.
    uint32_t gidx_begin = inputs.feature_segments[fidx];  // begining bin
    int split_gidx = (scan_begin + threadIdx.x) - 1;
    float fvalue;
    if (split_gidx < static_cast<int>(gidx_begin)) {
      fvalue = inputs.min_fvalue[fidx];
    } else {
      fvalue = inputs.feature_values[split_gidx];
    }
    GradientSumT left = missing_left ? bin + missing : bin;
    GradientSumT right = inputs.parent_sum - left;
    best_split->Update(gain, missing_left ? kLeftDir : kRightDir, fvalue,
                       fidx, GradientPair(left), GradientPair(right),
                       false, inputs.param);
  }
};

/*! \brief Find the thread with best gain. */
template <int BLOCK_THREADS, typename ReduceT, typename ScanT,
  typename MaxReduceT, typename TempStorageT, typename GradientSumT,
  typename BinFn, typename UpdateFn>
__device__ void EvaluateFeature(
    int fidx, EvaluateSplitInputs<GradientSumT> inputs,
    TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
    DeviceSplitCandidate* best_split,  // shared memory storing best split
    TempStorageT* temp_storage         // temp memory for cub operations
) {
  // Use pointer from cut to indicate begin and end of bins for each feature.
  uint32_t gidx_begin = inputs.feature_segments[fidx];  // begining bin
  uint32_t gidx_end =
      inputs.feature_segments[fidx + 1];  // end bin for i^th feature
  auto feature_hist = inputs.gradient_histogram.subspan(gidx_begin, gidx_end - gidx_begin);
  auto bin_fn = BinFn();
  auto update_fn = UpdateFn();

  // Sum histogram bins for current feature
  GradientSumT const feature_sum =
      ReduceFeature<BLOCK_THREADS, ReduceT, TempStorageT, GradientSumT>(
          feature_hist, temp_storage);

  GradientSumT const missing = inputs.parent_sum - feature_sum;
  float const null_gain = -std::numeric_limits<bst_float>::infinity();

  SumCallbackOp<GradientSumT> prefix_op = SumCallbackOp<GradientSumT>();
  for (int scan_begin = gidx_begin; scan_begin < gidx_end;
       scan_begin += BLOCK_THREADS) {
    bool thread_active = (scan_begin + threadIdx.x) < gidx_end;
    auto bin = bin_fn(thread_active, scan_begin, &prefix_op, missing, inputs, temp_storage);

    // Whether the gradient of missing values is put to the left side.
    bool missing_left = true;
    float gain = null_gain;
    if (thread_active) {
      gain = LossChangeMissing(bin, missing, inputs.parent_sum, inputs.param,
                               inputs.nidx,
                               fidx,
                               evaluator,
                               missing_left);
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

    // Best thread updates split
    if (threadIdx.x == block_max.key) {
      update_fn(missing_left, scan_begin, gain, fidx, missing, bin, inputs,
                best_split);
    }
    cub::CTA_SYNC();
  }
}

template <int BLOCK_THREADS, typename GradientSumT>
__global__ void EvaluateSplitsKernel(
    EvaluateSplitInputs<GradientSumT> left,
    EvaluateSplitInputs<GradientSumT> right,
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
    EvaluateFeature<BLOCK_THREADS, SumReduceT, BlockScanT, MaxReduceT,
                    TempStorage, GradientSumT,
                    OneHotBin<GradientSumT, TempStorage>,
                    UpdateOneHot<GradientSumT>>(fidx, inputs, evaluator, &best_split,
                                                &temp_storage);
  } else {
    EvaluateFeature<BLOCK_THREADS, SumReduceT, BlockScanT, MaxReduceT,
                    TempStorage, GradientSumT,
                    NumericBin<GradientSumT, TempStorage, BlockScanT>,
                    UpdateNumeric<GradientSumT>>(fidx, inputs, evaluator, &best_split,
                                                 &temp_storage);
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

template <typename GradientSumT>
void EvaluateSplits(common::Span<DeviceSplitCandidate> out_splits,
                    TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
                    EvaluateSplitInputs<GradientSumT> left,
                    EvaluateSplitInputs<GradientSumT> right) {
  size_t combined_num_features =
      left.feature_set.size() + right.feature_set.size();
  dh::TemporaryArray<DeviceSplitCandidate> feature_best_splits(
      combined_num_features);
  // One block for each feature
  uint32_t constexpr kBlockThreads = 256;
  dh::LaunchKernel {uint32_t(combined_num_features), kBlockThreads, 0}(
      EvaluateSplitsKernel<kBlockThreads, GradientSumT>, left, right, evaluator,
      dh::ToSpan(feature_best_splits));

  // Reduce to get best candidate for left and right child over all features
  auto reduce_offset =
      dh::MakeTransformIterator<size_t>(thrust::make_counting_iterator(0llu),
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
  cub::DeviceSegmentedReduce::Sum(nullptr, temp_storage_bytes,
                                  feature_best_splits.data(), out_splits.data(),
                                  num_segments, reduce_offset, reduce_offset + 1);
  dh::TemporaryArray<int8_t> temp(temp_storage_bytes);
  cub::DeviceSegmentedReduce::Sum(temp.data().get(), temp_storage_bytes,
                                  feature_best_splits.data(), out_splits.data(),
                                  num_segments, reduce_offset, reduce_offset + 1);
}

template <typename GradientSumT>
void EvaluateSingleSplit(common::Span<DeviceSplitCandidate> out_split,
                         TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
                         EvaluateSplitInputs<GradientSumT> input) {
  EvaluateSplits(out_split, evaluator, input, {});
}

template void EvaluateSplits<GradientPair>(
    common::Span<DeviceSplitCandidate> out_splits,
    TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
    EvaluateSplitInputs<GradientPair> left,
    EvaluateSplitInputs<GradientPair> right);
template void EvaluateSplits<GradientPairPrecise>(
    common::Span<DeviceSplitCandidate> out_splits,
    TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
    EvaluateSplitInputs<GradientPairPrecise> left,
    EvaluateSplitInputs<GradientPairPrecise> right);
template void EvaluateSingleSplit<GradientPair>(
    common::Span<DeviceSplitCandidate> out_split,
    TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
    EvaluateSplitInputs<GradientPair> input);
template void EvaluateSingleSplit<GradientPairPrecise>(
    common::Span<DeviceSplitCandidate> out_split,
    TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
    EvaluateSplitInputs<GradientPairPrecise> input);
}  // namespace tree
}  // namespace xgboost
