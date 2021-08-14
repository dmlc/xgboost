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
    uint32_t gidx_begin = inputs.feature_segments[fidx];  // beginning bin
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
  uint32_t gidx_begin = inputs.feature_segments[fidx];  // beginning bin
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
  auto l_n_features = left.feature_segments.empty() ? 0 : left.feature_segments.size() - 1;
  auto r_n_features = right.feature_segments.empty() ? 0 : right.feature_segments.size() - 1;
  if (!(r_n_features == 0 || l_n_features == r_n_features)) {
    throw std::runtime_error("Invariant violated");
  }

  // Handle empty (trivial) input
  if (l_n_features == 0 && r_n_features == 0) {
    dh::LaunchN(out_splits.size(), [=]XGBOOST_DEVICE(std::size_t idx) {
      out_splits[idx] = DeviceSplitCandidate{};
    });
    return;
  }

  auto out_scan = EvaluateSplitsFindOptimalSplitsViaScan(evaluator, left, right);
  auto d_out_scan = dh::ToSpan(out_scan);

  auto reduce_key = dh::MakeTransformIterator<int>(
      thrust::make_counting_iterator<bst_feature_t>(0),
      [=] __device__(bst_feature_t i) -> int {
        if (i < l_n_features) {
          return 0;  // left node
        } else {
          return 1;  // right node
        }
      });
  auto reduce_val = dh::MakeTransformIterator<DeviceSplitCandidate>(
      thrust::make_counting_iterator<std::size_t>(0),
      [d_out_scan] __device__(std::size_t i) {
        ScanComputedElem<GradientSumT> c = d_out_scan[i];
        return DeviceSplitCandidate{c.best_loss_chg, c.best_direction, c.best_findex,
                                    c.best_fvalue, c.is_cat, GradientPair{c.left_sum},
                                    GradientPair{c.right_sum}};
      });
  GPUTrainingParam param = left.param;
  thrust::reduce_by_key(
      thrust::device, reduce_key, reduce_key + static_cast<std::ptrdiff_t>(out_scan.size()),
      reduce_val, thrust::make_discard_iterator(), out_splits.data(),
      thrust::equal_to<int>{},
      [param] __device__(DeviceSplitCandidate l, DeviceSplitCandidate r) {
        l.Update(r, param);
        return l;
      });
}

template <typename GradientSumT>
void EvaluateSingleSplit(common::Span<DeviceSplitCandidate> out_split,
                         TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
                         EvaluateSplitInputs<GradientSumT> input) {
  EvaluateSplits(out_split, evaluator, input, {});
}

template <typename GradientSumT>
dh::device_vector<ScanComputedElem<GradientSumT>>
EvaluateSplitsFindOptimalSplitsViaScan(
    TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
    EvaluateSplitInputs<GradientSumT> left,
    EvaluateSplitInputs<GradientSumT> right) {
  auto l_n_features = left.feature_segments.empty() ? 0 : left.feature_segments.size() - 1;
  auto r_n_features = right.feature_segments.empty() ? 0 : right.feature_segments.size() - 1;
  if (!(r_n_features == 0 || l_n_features == r_n_features)) {
    throw std::runtime_error("Invariant violated");
  }

  uint64_t left_hist_size = static_cast<uint64_t>(left.gradient_histogram.size());
  auto map_to_left_right = [left_hist_size] __device__(uint64_t idx) {
    if (idx < left_hist_size) {
      // Left child node
      return EvaluateSplitsHistEntry{ChildNodeIndicator::kLeftChild, idx};
    } else {
      // Right child node
      return EvaluateSplitsHistEntry{ChildNodeIndicator::kRightChild, idx - left_hist_size};
    }
  };

  std::size_t size = left.gradient_histogram.size() + right.gradient_histogram.size();
  auto for_count_iter = thrust::make_counting_iterator<uint64_t>(0);
  auto for_loc_iter = dh::MakeTransformIterator<EvaluateSplitsHistEntry>(
      for_count_iter, map_to_left_right);
  auto rev_count_iter = thrust::make_reverse_iterator(
      thrust::make_counting_iterator<uint64_t>(0) + static_cast<std::ptrdiff_t>(size));
  auto rev_loc_iter = dh::MakeTransformIterator<EvaluateSplitsHistEntry>(
      rev_count_iter, map_to_left_right);
  auto zip_loc_iter = thrust::make_zip_iterator(thrust::make_tuple(for_loc_iter, rev_loc_iter));

  auto scan_input_iter =
      dh::MakeTransformIterator<thrust::tuple<ScanElem<GradientSumT>, ScanElem<GradientSumT>>>(
      zip_loc_iter, ScanValueOp<GradientSumT>{left, right, evaluator});

  dh::device_vector<ScanComputedElem<GradientSumT>> out_scan(l_n_features + r_n_features);
  auto scan_out_iter = thrust::make_transform_output_iterator(
      thrust::make_discard_iterator(),
      WriteScan<GradientSumT>{left, right, evaluator, dh::ToSpan(out_scan)});

  auto scan_op = ScanOp<GradientSumT>{left, right, evaluator};
  /*auto scan_op = []__device__(
      thrust::tuple<ScanElem<GradientSumT>, ScanElem<GradientSumT>> lhs,
      thrust::tuple<ScanElem<GradientSumT>, ScanElem<GradientSumT>> rhs) {
    return lhs;
  };*/
  std::size_t n_temp_bytes = 0;
  cub::DeviceScan::InclusiveScan(nullptr, n_temp_bytes, scan_input_iter, scan_out_iter,
                                 scan_op, size);
  dh::TemporaryArray<int8_t> temp(n_temp_bytes);
  cub::DeviceScan::InclusiveScan(temp.data().get(), n_temp_bytes, scan_input_iter, scan_out_iter,
                                 scan_op, size);
  return out_scan;
}

template <typename GradientSumT>
template <bool forward>
XGBOOST_DEVICE ScanElem<GradientSumT>
ScanValueOp<GradientSumT>::MapEvaluateSplitsHistEntryToScanElem(
    EvaluateSplitsHistEntry entry,
    EvaluateSplitInputs<GradientSumT> split_input) {
  ScanElem<GradientSumT> ret;
  ret.indicator = entry.indicator;
  ret.hist_idx = entry.hist_idx;
  ret.gpair = split_input.gradient_histogram[entry.hist_idx];
  ret.findex = static_cast<int32_t>(dh::SegmentId(split_input.feature_segments, entry.hist_idx));
  ret.fvalue = split_input.feature_values[entry.hist_idx];
  ret.is_cat = IsCat(split_input.feature_types, ret.findex);
  if ((forward && split_input.feature_segments[ret.findex] == entry.hist_idx) ||
      (!forward && split_input.feature_segments[ret.findex + 1] - 1 == entry.hist_idx)) {
    /**
     * For the element at the beginning of each segment, compute gradient sums and loss_chg
     * ahead of time. These will be later used by the inclusive scan operator.
     **/
    if (ret.is_cat) {
      ret.computed_result.left_sum = split_input.parent_sum - ret.gpair;
      ret.computed_result.right_sum = ret.gpair;
    } else {
      if (forward) {
        ret.computed_result.left_sum = ret.gpair;
        ret.computed_result.right_sum = split_input.parent_sum - ret.gpair;
      } else {
        ret.computed_result.left_sum = split_input.parent_sum - ret.gpair;
        ret.computed_result.right_sum = ret.gpair;
      }
    }
    ret.computed_result.parent_sum = split_input.parent_sum;
    float parent_gain = evaluator.CalcGain(split_input.nidx, split_input.param,
                                           GradStats{ret.computed_result.parent_sum});
    float gain = evaluator.CalcSplitGain(split_input.param, split_input.nidx, ret.findex,
                                         GradStats{ret.computed_result.left_sum},
                                         GradStats{ret.computed_result.right_sum});
    ret.computed_result.best_loss_chg = gain - parent_gain;
    ret.computed_result.best_findex = ret.findex;
    ret.computed_result.best_fvalue = ret.fvalue;
    ret.computed_result.best_direction =
        (forward ? DefaultDirection::kRightDir : DefaultDirection::kLeftDir);
  }

  return ret;
}

template <typename GradientSumT>
XGBOOST_DEVICE thrust::tuple<ScanElem<GradientSumT>, ScanElem<GradientSumT>>
ScanValueOp<GradientSumT>::operator() (
    thrust::tuple<EvaluateSplitsHistEntry, EvaluateSplitsHistEntry> entry_tup) {
  const auto& fw = thrust::get<0>(entry_tup);
  const auto& bw = thrust::get<1>(entry_tup);
  ScanElem<GradientSumT> ret_fw, ret_bw;
  ret_fw = MapEvaluateSplitsHistEntryToScanElem<true>(
      fw,
      (fw.indicator == ChildNodeIndicator::kLeftChild ? this->left : this->right));
  ret_bw = MapEvaluateSplitsHistEntryToScanElem<false>(
      bw,
      (bw.indicator == ChildNodeIndicator::kLeftChild ? this->left : this->right));
  return thrust::make_tuple(ret_fw, ret_bw);
}

template <typename GradientSumT>
template <bool forward>
XGBOOST_DEVICE ScanElem<GradientSumT>
ScanOp<GradientSumT>::DoIt(ScanElem<GradientSumT> lhs, ScanElem<GradientSumT> rhs) {
  ScanElem<GradientSumT> ret;
  ret = rhs;
  ret.computed_result = {};
  if (lhs.findex != rhs.findex || lhs.indicator != rhs.indicator) {
    // Segmented Scan
    return rhs;
  }
  if (((lhs.indicator == ChildNodeIndicator::kLeftChild) &&
      (left.feature_set.size() != left.feature_segments.size()) &&
      !thrust::binary_search(thrust::seq, left.feature_set.begin(), left.feature_set.end(),
                             lhs.findex)) ||
      ((lhs.indicator == ChildNodeIndicator::kRightChild) &&
      (right.feature_set.size() != right.feature_segments.size()) &&
      !thrust::binary_search(thrust::seq, right.feature_set.begin(), right.feature_set.end(),
                             lhs.findex))) {
    // Column sampling
    return rhs;
  }

  GradientSumT parent_sum = lhs.computed_result.parent_sum;
  GradientSumT left_sum, right_sum;
  if (lhs.is_cat) {
    left_sum = lhs.computed_result.parent_sum - rhs.gpair;
    right_sum = rhs.gpair;
  } else {
    if (forward) {
      left_sum = lhs.computed_result.left_sum + rhs.gpair;
      right_sum = lhs.computed_result.parent_sum - left_sum;
    } else {
      right_sum = lhs.computed_result.right_sum + rhs.gpair;
      left_sum = lhs.computed_result.parent_sum - right_sum;
    }
  }
  bst_node_t nidx = (lhs.indicator == ChildNodeIndicator::kLeftChild) ? left.nidx : right.nidx;
  float gain = evaluator.CalcSplitGain(
      left.param, nidx, lhs.findex, GradStats{left_sum}, GradStats{right_sum});
  float parent_gain = evaluator.CalcGain(left.nidx, left.param, GradStats{parent_sum});
  float loss_chg = gain - parent_gain;
  ret.computed_result = lhs.computed_result;
  ret.computed_result.Update(left_sum, right_sum, parent_sum,
                             loss_chg, lhs.findex, lhs.is_cat, lhs.fvalue,
                             (forward ? DefaultDirection::kRightDir : DefaultDirection::kLeftDir),
                             left.param);
  return ret;
}

template <typename GradientSumT>
XGBOOST_DEVICE thrust::tuple<ScanElem<GradientSumT>, ScanElem<GradientSumT>>
ScanOp<GradientSumT>::operator() (
    thrust::tuple<ScanElem<GradientSumT>, ScanElem<GradientSumT>> lhs,
    thrust::tuple<ScanElem<GradientSumT>, ScanElem<GradientSumT>> rhs) {
  const auto& lhs_fw = thrust::get<0>(lhs);
  const auto& lhs_bw = thrust::get<1>(lhs);
  const auto& rhs_fw = thrust::get<0>(rhs);
  const auto& rhs_bw = thrust::get<1>(rhs);
  //return lhs;
  return thrust::make_tuple(DoIt<true>(lhs_fw, rhs_fw), DoIt<false>(lhs_bw, rhs_bw));
};

template <typename GradientSumT>
template <bool forward>
void
XGBOOST_DEVICE WriteScan<GradientSumT>::DoIt(ScanElem<GradientSumT> e) {
  EvaluateSplitInputs<GradientSumT>& split_input =
      (e.indicator == ChildNodeIndicator::kLeftChild) ? left : right;
  std::size_t offset = 0;
  std::size_t n_features = left.feature_segments.empty() ? 0 : left.feature_segments.size() - 1;
  if (e.indicator == ChildNodeIndicator::kRightChild) {
    offset = n_features;
  }
  if ((!forward && split_input.feature_segments[e.findex] == e.hist_idx) ||
      (forward && split_input.feature_segments[e.findex + 1] - 1 == e.hist_idx)) {
    if (e.computed_result.best_loss_chg > d_out_scan[offset + e.findex].best_loss_chg) {
      d_out_scan[offset + e.findex] = e.computed_result;
    }
  }
}

template <typename GradientSumT>
thrust::tuple<ScanElem<GradientSumT>, ScanElem<GradientSumT>>
XGBOOST_DEVICE WriteScan<GradientSumT>::operator() (
    thrust::tuple<ScanElem<GradientSumT>, ScanElem<GradientSumT>> e) {
  const auto& fw = thrust::get<0>(e);
  const auto& bw = thrust::get<1>(e);
  DoIt<true>(fw);
  DoIt<false>(bw);
  return {};  // discard
}

template <typename GradientSumT>
XGBOOST_DEVICE bool
ScanComputedElem<GradientSumT>::Update(
    GradientSumT left_sum_in,
    GradientSumT right_sum_in,
    GradientSumT parent_sum_in,
    float loss_chg_in,
    int32_t findex_in,
    bool is_cat_in,
    float fvalue_in,
    DefaultDirection dir_in,
    const GPUTrainingParam& param) {
  if (loss_chg_in > best_loss_chg &&
      left_sum_in.GetHess() >= param.min_child_weight &&
      right_sum_in.GetHess() >= param.min_child_weight) {
    best_loss_chg = loss_chg_in;
    best_findex = findex_in;
    is_cat = is_cat_in;
    best_fvalue = fvalue_in;
    best_direction = dir_in;
    left_sum = left_sum_in;
    right_sum = right_sum_in;
    parent_sum = parent_sum_in;
    return true;
  }
  return false;
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
