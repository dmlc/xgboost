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

  auto out_scan = EvaluateSplitsGenerateSplitCandidatesViaScan(evaluator, left, right);
  auto d_out_scan = dh::ToSpan(out_scan);

  auto reduce_key = dh::MakeTransformIterator<int>(
      thrust::make_counting_iterator<std::size_t>(0),
      [d_out_scan] __device__(std::size_t i) {
        return d_out_scan[i].node_idx;
      });
  auto reduce_val = dh::MakeTransformIterator<DeviceSplitCandidate>(
      thrust::make_counting_iterator<std::size_t>(0),
      [d_out_scan] __device__(std::size_t i) {
        const auto& e = d_out_scan[i];
        GradientSumT left_sum, right_sum;
        if (e.is_cat) {
          left_sum = e.parent_sum - e.partial_sum;
          right_sum = e.partial_sum;
        } else {
          if (e.direction == DefaultDirection::kRightDir) {
            left_sum = e.partial_sum;
            right_sum = e.parent_sum - e.partial_sum;
          } else {
            left_sum = e.parent_sum - e.partial_sum;
            right_sum = e.partial_sum;
          }
        }
        return DeviceSplitCandidate{e.loss_chg, e.direction, e.findex, e.fvalue, e.is_cat,
                                    GradientPair{left_sum}, GradientPair{right_sum}};
      });
  /**
   * Perform segmented reduce to find the best split candidate per node.
   * Note that there will be THREE segments:
   * [segment for left child node] [segment for right child node] [segment for left child node]
   * This is due to how we perform forward and backward passes over the gradient histogram.
   */
  dh::device_vector<DeviceSplitCandidate> out_reduce(3);
  GPUTrainingParam param = left.param;
  thrust::reduce_by_key(
      thrust::device, reduce_key, reduce_key + static_cast<std::ptrdiff_t>(out_scan.size()),
      reduce_val, thrust::make_discard_iterator(), out_reduce.data(),
      thrust::equal_to<int>{},
      [param] __device__(DeviceSplitCandidate l, DeviceSplitCandidate r) {
        l.Update(r, param);
        return l;
      });
  if (right.gradient_histogram.empty()) {
    dh::LaunchN(1, [out_reduce = dh::ToSpan(out_reduce), out_splits]__device__(std::size_t) {
      out_splits[0] = out_reduce[0];
    });
  } else {
    dh::LaunchN(1, [out_reduce = dh::ToSpan(out_reduce), out_splits, param]__device__(std::size_t) {
      out_reduce[0].Update(out_reduce[2], param);
      out_splits[0] = out_reduce[0];
      out_splits[1] = out_reduce[1];
    });
  }
}

template <typename GradientSumT>
void EvaluateSingleSplit(common::Span<DeviceSplitCandidate> out_split,
                         TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
                         EvaluateSplitInputs<GradientSumT> input) {
  EvaluateSplits(out_split, evaluator, input, {});
}

template <typename GradientSumT>
dh::device_vector<ReduceElem<GradientSumT>>
EvaluateSplitsGenerateSplitCandidatesViaScan(
    TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
    EvaluateSplitInputs<GradientSumT> left,
    EvaluateSplitInputs<GradientSumT> right) {
  auto l_n_features = left.feature_segments.empty() ? 0 : left.feature_segments.size() - 1;
  auto r_n_features = right.feature_segments.empty() ? 0 : right.feature_segments.size() - 1;
  if (!(r_n_features == 0 || l_n_features == r_n_features)) {
    throw std::runtime_error("Invariant violated");
  }

  std::size_t size = left.gradient_histogram.size() + right.gradient_histogram.size();
  CHECK_LE(size, static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()));
  auto count_iter = dh::MakeTransformIterator<thrust::tuple<uint32_t, bool>>(
      thrust::make_counting_iterator<uint32_t>(0),
      [size = static_cast<uint32_t>(size)] __device__(uint32_t i) {
        // Generate sequence of length (size * 2):
        // 0 1 2 3 ... (size-2) (size-1) (size-1) (size-2) ... 2 1 0
        // At each element, attach a boolean indicating whether the sequence is going forward or
        // backward.
        if (i < size) {
          return thrust::make_tuple(i, true);
        } else if (i < size * 2) {
          // size <= size < size * 2
          return thrust::make_tuple(size * 2 - 1 - i, false);
        } else {
          return thrust::make_tuple(static_cast<uint32_t>(0), false);
            // out-of-bounds, just return 0
        }
      });

  CHECK_LE(left.gradient_histogram.size(),
           static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()));
  uint32_t left_hist_size = static_cast<uint32_t>(left.gradient_histogram.size());
  auto map_to_hist_bin = [left_hist_size] __device__(thrust::tuple<uint32_t, bool> e) {
    // The first (size) outputs will be of the forward pass
    // The following (size) outputs will be of the backward pass
    uint32_t idx = thrust::get<0>(e);
    bool forward = thrust::get<1>(e);
    if (idx < left_hist_size) {
      // Left child node
      return EvaluateSplitsHistEntry{0, idx, forward};
    } else {
      // Right child node
      return EvaluateSplitsHistEntry{1, idx - left_hist_size, forward};
    }
  };  // NOLINT (readability/braces)
  auto bin_iter = dh::MakeTransformIterator<EvaluateSplitsHistEntry>(count_iter, map_to_hist_bin);
  auto scan_input_iter =
      dh::MakeTransformIterator<ScanElem<GradientSumT>>(
          bin_iter, ScanValueOp<GradientSumT>{left, right, evaluator});

  dh::device_vector<ReduceElem<GradientSumT>> out_scan(size * 2);
  auto scan_out_iter = thrust::make_transform_output_iterator(
      out_scan.begin(), ReduceValueOp<GradientSumT>{left, right, evaluator});

  auto scan_op = ScanOp<GradientSumT>{left, right, evaluator};
  std::size_t n_temp_bytes = 0;
  cub::DeviceScan::InclusiveScan(nullptr, n_temp_bytes, scan_input_iter, scan_out_iter,
                                 scan_op, size * 2);
  dh::TemporaryArray<int8_t> temp(n_temp_bytes);
  cub::DeviceScan::InclusiveScan(temp.data().get(), n_temp_bytes, scan_input_iter, scan_out_iter,
                                 scan_op, size * 2);
  return out_scan;
}

template <typename GradientSumT>
__device__ ScanElem<GradientSumT>
ScanValueOp<GradientSumT>::MapEvaluateSplitsHistEntryToScanElem(
    EvaluateSplitsHistEntry entry,
    EvaluateSplitInputs<GradientSumT> split_input) {
  ScanElem<GradientSumT> ret;
  ret.node_idx = entry.node_idx;
  ret.hist_idx = entry.hist_idx;
  ret.findex = static_cast<int32_t>(dh::SegmentId(split_input.feature_segments, entry.hist_idx));
  if (entry.forward) {
    ret.fvalue = split_input.feature_values[entry.hist_idx];
  } else {
    if (entry.hist_idx > 0) {
      ret.fvalue = split_input.feature_values[entry.hist_idx - 1];
    } else {
      ret.fvalue = split_input.min_fvalue[ret.findex];
    }
  }
  ret.is_cat = IsCat(split_input.feature_types, ret.findex);
  ret.forward = entry.forward;
  ret.gpair = split_input.gradient_histogram[entry.hist_idx];
  ret.parent_sum = split_input.parent_sum;
  if (((entry.node_idx == 0) &&
       (left.feature_set.size() != left.feature_segments.size()) &&
       !thrust::binary_search(thrust::seq, left.feature_set.begin(),
                              left.feature_set.end(), ret.findex)) ||
      ((entry.node_idx == 1) &&
       (right.feature_set.size() != right.feature_segments.size()) &&
       !thrust::binary_search(thrust::seq, right.feature_set.begin(),
                              right.feature_set.end(), ret.findex))) {
    // Column sampling
    return ret;
  }
  ret.partial_sum = ret.gpair;

  return ret;
}

template <typename GradientSumT>
__device__ ScanElem<GradientSumT>
ScanValueOp<GradientSumT>::operator() (EvaluateSplitsHistEntry entry) {
  return MapEvaluateSplitsHistEntryToScanElem(
      entry, (entry.node_idx == 0 ? this->left : this->right));
}

template <typename GradientSumT>
__device__ ScanElem<GradientSumT>
ScanOp<GradientSumT>::DoIt(ScanElem<GradientSumT> lhs, ScanElem<GradientSumT> rhs) {
  ScanElem<GradientSumT> ret;
  if (lhs.findex != rhs.findex || lhs.node_idx != rhs.node_idx || lhs.forward != rhs.forward) {
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

  ret = rhs;
  ret.partial_sum = lhs.partial_sum + rhs.partial_sum;
  return ret;
}

template <typename GradientSumT>
__device__ ScanElem<GradientSumT>
ScanOp<GradientSumT>::operator() (ScanElem<GradientSumT> lhs, ScanElem<GradientSumT> rhs) {
  return DoIt(lhs, rhs);
}

template <typename GradientSumT>
ReduceElem<GradientSumT>
__device__ ReduceValueOp<GradientSumT>::DoIt(ScanElem<GradientSumT> e) {
  ReduceElem<GradientSumT> ret;
  if (e.is_cat) {
    ret.partial_sum = e.gpair;
  } else {
    ret.partial_sum = e.partial_sum;
  }
  ret.parent_sum = e.parent_sum;
  {
    GradientSumT left_sum, right_sum;
    if (e.is_cat) {
      left_sum = e.parent_sum - e.gpair;
      right_sum = e.gpair;
    } else {
      if (e.forward) {
        left_sum = e.partial_sum;
        right_sum = e.parent_sum - e.partial_sum;
      } else {
        left_sum = e.parent_sum - e.partial_sum;
        right_sum = e.partial_sum;
      }
    }
    if (left_sum.GetHess() >= left.param.min_child_weight
        && right_sum.GetHess() >= left.param.min_child_weight) {
      // Enforce min_child_weight constraint
      bst_node_t nidx = (e.node_idx == 0) ? left.nidx : right.nidx;
      float gain = evaluator.CalcSplitGain(left.param, nidx, e.findex, GradStats{left_sum},
                                           GradStats{right_sum});
      float parent_gain = evaluator.CalcGain(left.nidx, left.param, GradStats{e.parent_sum});
      ret.loss_chg = gain - parent_gain;
    } else {
      ret.loss_chg = std::numeric_limits<float>::lowest();
    }
  }
  ret.findex = e.findex;
  ret.node_idx = e.node_idx;
  ret.fvalue = e.fvalue;
  ret.is_cat = e.is_cat;
  ret.direction = (e.forward ? DefaultDirection::kRightDir : DefaultDirection::kLeftDir);
  return ret;
}

template <typename GradientSumT>
ReduceElem<GradientSumT>
__device__ ReduceValueOp<GradientSumT>::operator() (ScanElem<GradientSumT> e) {
  return DoIt(e);
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
