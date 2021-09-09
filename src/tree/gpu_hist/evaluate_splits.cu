/*!
 * Copyright 2020 by XGBoost Contributors
 */
#include <limits>
#include "evaluate_splits.cuh"
#include "../../common/categorical.h"

namespace xgboost {
namespace tree {

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
    dh::LaunchN(out_splits.size(), [=]__device__(std::size_t idx) {
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
        GradientSumT left_sum, right_sum;
        if (c.is_cat) {
          left_sum = c.parent_sum - c.best_partial_sum;
          right_sum = c.best_partial_sum;
        } else {
          if (c.best_direction == DefaultDirection::kRightDir) {
            left_sum = c.best_partial_sum;
            right_sum = c.parent_sum - c.best_partial_sum;
          } else {
            left_sum = c.parent_sum - c.best_partial_sum;
            right_sum = c.best_partial_sum;
          }
        }
        return DeviceSplitCandidate{c.best_loss_chg, c.best_direction, c.best_findex,
                                    c.best_fvalue, c.is_cat, GradientPair{left_sum},
                                    GradientPair{right_sum}};
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

  uint32_t left_hist_size = static_cast<uint32_t>(left.gradient_histogram.size());
  auto map_to_hist_bin = [left_hist_size] __device__(uint32_t idx) {
    if (idx < left_hist_size) {
      // Left child node
      return EvaluateSplitsHistEntry{0, idx};
    } else {
      // Right child node
      return EvaluateSplitsHistEntry{1, idx - left_hist_size};
    }
  };

  std::size_t size = left.gradient_histogram.size() + right.gradient_histogram.size();
  auto forward_count_iter = thrust::make_counting_iterator<uint32_t>(0);
  auto forward_bin_iter = dh::MakeTransformIterator<EvaluateSplitsHistEntry>(
      forward_count_iter, map_to_hist_bin);
  auto forward_scan_input_iter = dh::MakeTransformIterator<ScanElem<GradientSumT>>(
      forward_bin_iter, ScanValueOp<GradientSumT>{true, left, right, evaluator});

  dh::device_vector<ScanComputedElem<GradientSumT>> out_scan(l_n_features + r_n_features);
  auto forward_scan_out_iter = thrust::make_transform_output_iterator(
      thrust::make_discard_iterator(),
      WriteScan<GradientSumT>{true, left, right, evaluator, dh::ToSpan(out_scan)});
  {
    auto scan_op = ScanOp<GradientSumT>{true, left, right, evaluator};
    std::size_t n_temp_bytes = 0;
    cub::DeviceScan::InclusiveScan(nullptr, n_temp_bytes, forward_scan_input_iter,
                                   forward_scan_out_iter, scan_op, size);
    dh::TemporaryArray<int8_t> temp(n_temp_bytes);
    cub::DeviceScan::InclusiveScan(temp.data().get(), n_temp_bytes, forward_scan_input_iter,
                                   forward_scan_out_iter, scan_op, size);
  }

  auto backward_count_iter = thrust::make_reverse_iterator(
      thrust::make_counting_iterator<uint32_t>(0) + static_cast<std::ptrdiff_t>(size));
  auto backward_bin_iter = dh::MakeTransformIterator<EvaluateSplitsHistEntry>(
      backward_count_iter, map_to_hist_bin);
  auto backward_scan_input_iter = dh::MakeTransformIterator<ScanElem<GradientSumT>>(
      backward_bin_iter, ScanValueOp<GradientSumT>{false, left, right, evaluator});
  auto backward_scan_out_iter = thrust::make_transform_output_iterator(
      thrust::make_discard_iterator(),
      WriteScan<GradientSumT>{false, left, right, evaluator, dh::ToSpan(out_scan)});
  {
    auto scan_op = ScanOp<GradientSumT>{false, left, right, evaluator};
    std::size_t n_temp_bytes = 0;
    cub::DeviceScan::InclusiveScan(nullptr, n_temp_bytes, backward_scan_input_iter,
                                   backward_scan_out_iter, scan_op, size);
    dh::TemporaryArray<int8_t> temp(n_temp_bytes);
    cub::DeviceScan::InclusiveScan(temp.data().get(), n_temp_bytes, backward_scan_input_iter,
                                   backward_scan_out_iter, scan_op, size);
  }

  return out_scan;
}

template <typename GradientSumT>
__noinline__ __device__ ScanElem<GradientSumT>
ScanValueOp<GradientSumT>::MapEvaluateSplitsHistEntryToScanElem(
    EvaluateSplitsHistEntry entry,
    EvaluateSplitInputs<GradientSumT> split_input) {
  ScanElem<GradientSumT> ret;
  ret.node_idx = entry.node_idx;
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
    GradientSumT partial_sum = ret.gpair;
    GradientSumT complement_sum = split_input.parent_sum - partial_sum;
    GradientSumT *left_sum, *right_sum;
    if (ret.is_cat) {
      left_sum = &complement_sum;
      right_sum = &partial_sum;
    } else {
      if (forward) {
        left_sum = &partial_sum;
        right_sum = &complement_sum;
      } else {
        left_sum = &complement_sum;
        right_sum = &partial_sum;
      }
    }
    ret.computed_result.parent_sum = partial_sum;
    ret.computed_result.best_partial_sum = partial_sum;
    ret.computed_result.parent_sum = split_input.parent_sum;
    float parent_gain = evaluator.CalcGain(split_input.nidx, split_input.param,
                                           GradStats{split_input.parent_sum});
    float gain = evaluator.CalcSplitGain(split_input.param, split_input.nidx, ret.findex,
                                         GradStats{*left_sum}, GradStats{*right_sum});
    ret.computed_result.best_loss_chg = gain - parent_gain;
    ret.computed_result.best_findex = ret.findex;
    ret.computed_result.best_fvalue = ret.fvalue;
    ret.computed_result.best_direction =
        (forward ? DefaultDirection::kRightDir : DefaultDirection::kLeftDir);
    ret.computed_result.is_cat = ret.is_cat;
  }

  return ret;
}

template <typename GradientSumT>
__noinline__ __device__ ScanElem<GradientSumT>
ScanValueOp<GradientSumT>::operator() (EvaluateSplitsHistEntry entry) {
  return MapEvaluateSplitsHistEntryToScanElem(
      entry, (entry.node_idx == 0 ? this->left : this->right));
}

template <typename GradientSumT>
__noinline__ __device__ ScanElem<GradientSumT>
ScanOp<GradientSumT>::DoIt(ScanElem<GradientSumT> lhs, ScanElem<GradientSumT> rhs) {
  ScanElem<GradientSumT> ret;
  ret = rhs;
  ret.computed_result = {};
  if (lhs.findex != rhs.findex || lhs.node_idx != rhs.node_idx) {
    // Segmented Scan
    return rhs;
  }
  if (((lhs.node_idx == 0) &&
      (left.feature_set.size() != left.feature_segments.size()) &&
      !thrust::binary_search(thrust::seq, left.feature_set.begin(), left.feature_set.end(),
                             lhs.findex)) ||
      ((lhs.node_idx == 1) &&
      (right.feature_set.size() != right.feature_segments.size()) &&
      !thrust::binary_search(thrust::seq, right.feature_set.begin(), right.feature_set.end(),
                             lhs.findex))) {
    // Column sampling
    return rhs;
  }

  GradientSumT parent_sum = lhs.computed_result.parent_sum;
  GradientSumT partial_sum, complement_sum;
  GradientSumT *left_sum, *right_sum;
  if (lhs.is_cat) {
    partial_sum = rhs.gpair;
    complement_sum = lhs.computed_result.parent_sum - rhs.gpair;
    left_sum = &complement_sum;
    right_sum = &partial_sum;
  } else {
    partial_sum = lhs.computed_result.partial_sum + rhs.gpair;
    complement_sum = parent_sum - partial_sum;
    if (forward) {
      left_sum = &partial_sum;
      right_sum = &complement_sum;
    } else {
      left_sum = &complement_sum;
      right_sum = &partial_sum;
    }
  }
  bst_node_t nidx = (lhs.node_idx == 0) ? left.nidx : right.nidx;
  float gain = evaluator.CalcSplitGain(
      left.param, nidx, lhs.findex, GradStats{*left_sum}, GradStats{*right_sum});
  float parent_gain = evaluator.CalcGain(left.nidx, left.param, GradStats{parent_sum});
  float loss_chg = gain - parent_gain;
  ret.computed_result = lhs.computed_result;
  ret.computed_result.Update(partial_sum, parent_sum, loss_chg, rhs.findex, rhs.is_cat, rhs.fvalue,
                             (forward ? DefaultDirection::kRightDir : DefaultDirection::kLeftDir),
                             left.param);
  return ret;
}

template <typename GradientSumT>
__noinline__ __device__ ScanElem<GradientSumT>
ScanOp<GradientSumT>::operator() (ScanElem<GradientSumT> lhs, ScanElem<GradientSumT> rhs) {
  return DoIt(lhs, rhs);
};

template <typename GradientSumT>
void
__noinline__ __device__ WriteScan<GradientSumT>::DoIt(ScanElem<GradientSumT> e) {
  EvaluateSplitInputs<GradientSumT>& split_input = (e.node_idx == 0) ? left : right;
  std::size_t offset = 0;
  std::size_t n_features = left.feature_segments.empty() ? 0 : left.feature_segments.size() - 1;
  if (e.node_idx == 1) {
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
ScanElem<GradientSumT>
__noinline__ __device__ WriteScan<GradientSumT>::operator() (ScanElem<GradientSumT> e) {
  DoIt(e);
  return {};  // discard
}

template <typename GradientSumT>
__noinline__ __device__ bool
ScanComputedElem<GradientSumT>::Update(
    GradientSumT partial_sum_in,
    GradientSumT parent_sum_in,
    float loss_chg_in,
    int32_t findex_in,
    bool is_cat_in,
    float fvalue_in,
    DefaultDirection dir_in,
    const GPUTrainingParam& param) {
  partial_sum = partial_sum_in;
  parent_sum = parent_sum_in;
  if (loss_chg_in > best_loss_chg &&
      partial_sum_in.GetHess() >= param.min_child_weight &&
      (parent_sum_in.GetHess() - partial_sum_in.GetHess()) >= param.min_child_weight) {
    best_loss_chg = loss_chg_in;
    best_findex = findex_in;
    is_cat = is_cat_in;
    best_fvalue = fvalue_in;
    best_direction = dir_in;
    best_partial_sum = partial_sum_in;
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
