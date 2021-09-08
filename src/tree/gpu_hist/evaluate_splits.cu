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
        return DeviceSplitCandidate{c.best_loss_chg, c.best_direction, c.best_findex,
                                    c.best_fvalue, c.is_cat, GradientPair{c.best_left_sum},
                                    GradientPair{c.best_right_sum}};
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
  {
    auto write_computed_result = [](std::ostream& os, const ScanComputedElem<GradientSumT>& m) {
      std::string best_direction_str =
          (m.best_direction == DefaultDirection::kLeftDir) ? "left" : "right";
      os << "(left_sum: " << m.left_sum << ", right_sum: " << m.right_sum
         << ", parent_sum: " << m.parent_sum << ", best_loss_chg: " << m.best_loss_chg
         << ", best_findex: " << m.best_findex << ", best_fvalue: " << m.best_fvalue
         << ", best_direction: " << best_direction_str << ")";
    };
    auto write_scan_elem = [&](std::ostream& os, const ScanElem<GradientSumT>& m) {
      std::string indicator_str =
          (m.indicator == ChildNodeIndicator::kLeftChild) ? "kLeftChild" : "kRightChild";
      os << "(head_flag: " << (m.head_flag ? "true" : "false") << ", indicator: " << indicator_str
         << ", hist_idx: " << m.hist_idx << ", findex: " << m.findex << ", gpair: "<< m.gpair
         << ", fvalue: " << m.fvalue << ", is_cat: " << (m.is_cat ? "true" : "false")
         << ", computed_result: ";
      write_computed_result(os, m.computed_result);
      os << ")";
    };
    {
      using TupT = thrust::tuple<ScanElem<GradientSumT>, ScanElem<GradientSumT>>;
      thrust::device_vector<TupT> d_vec(size);
      thrust::host_vector<TupT> vec(size);
      thrust::copy(thrust::device, scan_input_iter, scan_input_iter + size, d_vec.begin());
      thrust::copy(d_vec.begin(), d_vec.end(), vec.begin());
      std::ostringstream oss;
      for (const auto& e: vec) {
        auto fw = thrust::get<0>(e);
        auto bw = thrust::get<1>(e);
        oss << "forward: ";
        write_scan_elem(oss, fw);
        oss << std::endl;
        oss << "backward: ";
        write_scan_elem(oss, bw);
        oss << std::endl;
      }
      LOG(CONSOLE) << oss.str();
    }

    {
      using TupT = thrust::tuple<ScanElem<GradientSumT>, ScanElem<GradientSumT>>;
      thrust::device_vector<TupT> d_vec(size);

      auto scan_op = ScanOp<GradientSumT>{left, right, evaluator};
      std::size_t n_temp_bytes = 0;
      cub::DeviceScan::InclusiveScan(nullptr, n_temp_bytes, scan_input_iter, d_vec.begin(),
                                     scan_op, size);
      dh::TemporaryArray<int8_t> temp(n_temp_bytes);
      cub::DeviceScan::InclusiveScan(temp.data().get(), n_temp_bytes, scan_input_iter,
                                     d_vec.begin(),
                                     scan_op, size);

      thrust::host_vector<TupT> vec(size);
      thrust::copy(d_vec.begin(), d_vec.end(), vec.begin());
      std::ostringstream oss;
      for (const auto& e: vec) {
        auto fw = thrust::get<0>(e);
        auto bw = thrust::get<1>(e);
        oss << "forward: ";
        write_scan_elem(oss, fw);
        oss << std::endl;
        oss << "backward: ";
        write_scan_elem(oss, bw);
        oss << std::endl;
      }
      LOG(CONSOLE) << oss.str();
    }
  }

  dh::device_vector<ScanComputedElem<GradientSumT>> out_scan(l_n_features + r_n_features);
  auto scan_out_iter = thrust::make_transform_output_iterator(
      thrust::make_discard_iterator(),
      WriteScan<GradientSumT>{left, right, evaluator, dh::ToSpan(out_scan)});

  auto scan_op = ScanOp<GradientSumT>{left, right, evaluator};
  std::size_t n_temp_bytes = 0;
  cub::DeviceScan::InclusiveScan(nullptr, n_temp_bytes, scan_input_iter, scan_out_iter,
                                 scan_op, size);
  dh::TemporaryArray<int8_t> temp(n_temp_bytes);
  cub::DeviceScan::InclusiveScan(temp.data().get(), n_temp_bytes, scan_input_iter, scan_out_iter,
                                 scan_op, size);
  {
    auto write_computed_result = [](std::ostream& os, const ScanComputedElem<GradientSumT>& m) {
      std::string best_direction_str =
          (m.best_direction == DefaultDirection::kLeftDir) ? "left" : "right";
      os << "(left_sum: " << m.left_sum << ", right_sum: " << m.right_sum
         << ", parent_sum: " << m.parent_sum << ", best_loss_chg: " << m.best_loss_chg
         << ", best_findex: " << m.best_findex << ", best_fvalue: " << m.best_fvalue
         << ", best_direction: " << best_direction_str << ")";
    };
    thrust::host_vector<ScanComputedElem<GradientSumT>> h_out_scan(l_n_features + r_n_features);
    thrust::copy(out_scan.begin(), out_scan.end(), h_out_scan.begin());
    std::ostringstream oss;
    for (const auto& e : h_out_scan) {
      write_computed_result(oss, e);
      oss << std::endl;
    }
    LOG(CONSOLE) << oss.str();
  }
  return out_scan;
}

template <typename GradientSumT>
template <bool forward>
__noinline__ __device__ ScanElem<GradientSumT>
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
    ret.head_flag = true;
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
    ret.computed_result.best_left_sum = ret.computed_result.left_sum;
    ret.computed_result.best_right_sum = ret.computed_result.right_sum;
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
  } else {
    ret.head_flag = false;
  }

  return ret;
}

template <typename GradientSumT>
__noinline__ __device__ thrust::tuple<ScanElem<GradientSumT>, ScanElem<GradientSumT>>
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
__noinline__ __device__ ScanElem<GradientSumT>
ScanOp<GradientSumT>::DoIt(ScanElem<GradientSumT> lhs, ScanElem<GradientSumT> rhs) {
  if (rhs.head_flag) {
    // Segmented Scan
    return rhs;
  }
  ScanElem<GradientSumT> ret;
  ret = rhs;
  ret.head_flag = (lhs.head_flag || rhs.head_flag);
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
                             loss_chg, rhs.findex, rhs.is_cat, rhs.fvalue,
                             (forward ? DefaultDirection::kRightDir : DefaultDirection::kLeftDir),
                             left.param);
  return ret;
}

template <typename GradientSumT>
__noinline__ __device__ thrust::tuple<ScanElem<GradientSumT>, ScanElem<GradientSumT>>
ScanOp<GradientSumT>::operator() (
    thrust::tuple<ScanElem<GradientSumT>, ScanElem<GradientSumT>> lhs,
    thrust::tuple<ScanElem<GradientSumT>, ScanElem<GradientSumT>> rhs) {
  const auto& lhs_fw = thrust::get<0>(lhs);
  const auto& lhs_bw = thrust::get<1>(lhs);
  const auto& rhs_fw = thrust::get<0>(rhs);
  const auto& rhs_bw = thrust::get<1>(rhs);
  return thrust::make_tuple(DoIt<true>(lhs_fw, rhs_fw), DoIt<false>(lhs_bw, rhs_bw));
};

template <typename GradientSumT>
template <bool forward>
void
__noinline__ __device__ WriteScan<GradientSumT>::DoIt(ScanElem<GradientSumT> e) {
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
__noinline__ __device__ WriteScan<GradientSumT>::operator() (
    thrust::tuple<ScanElem<GradientSumT>, ScanElem<GradientSumT>> e) {
  const auto& fw = thrust::get<0>(e);
  const auto& bw = thrust::get<1>(e);
  DoIt<true>(fw);
  DoIt<false>(bw);
  return {};  // discard
}

template <typename GradientSumT>
__noinline__ __device__ bool
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
  left_sum = left_sum_in;
  right_sum = right_sum_in;
  parent_sum = parent_sum_in;
  if (loss_chg_in > best_loss_chg &&
      left_sum_in.GetHess() >= param.min_child_weight &&
      right_sum_in.GetHess() >= param.min_child_weight) {
    best_loss_chg = loss_chg_in;
    best_findex = findex_in;
    is_cat = is_cat_in;
    best_fvalue = fvalue_in;
    best_direction = dir_in;
    best_left_sum = left_sum_in;
    best_right_sum = right_sum_in;
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
