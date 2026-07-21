/**
 * Copyright 2025-2026, XGBoost contributors
 */
#include <thrust/logical.h>  // for any_of
#include <thrust/reduce.h>   // for reduce_by_key, reduce
#include <thrust/sort.h>     // for stable_sort_by_key

#include <cub/block/block_scan.cuh>  // for BlockScan
#include <cub/util_type.cuh>         // for KeyValuePair
#include <cub/warp/warp_reduce.cuh>  // for WarpReduce
#include <cuda/ptx>                  // for get_sreg_laneid
#include <cuda/std/functional>       // for identity
#include <cuda/std/tuple>            // for get, make_tuple, tuple
#include <limits>
#include <type_traits>  // for is_trivially_copyable_v
#include <vector>       // for vector

#include "../../common/cuda_context.cuh"
#include "../tree_view.h"             // for MultiTargetTreeView
#include "multi_evaluate_splits.cuh"  // for MultiEvalauteSplitInputs, MultiEvaluateSplitSharedInputs
#include "quantiser.cuh"              // for GradientQuantiser
#include "xgboost/base.h"             // for GradientPairInt64
#include "xgboost/span.h"             // for Span

namespace xgboost::tree::cuda_impl {
namespace {
/**
 * @brief Calculate the gradient index for the reverse pass
 *
 * @note All inputs are global across features.
 */
__device__ bst_bin_t RevBinIdx(bst_bin_t gidx_begin, bst_bin_t gidx_end, bst_bin_t bin_idx) {
  return gidx_begin + (gidx_end - bin_idx - 1);
}

// Scan the histogram in 2 dim for all nodes
struct ScanHistogramAgent {
  using WarpScanT = cub::WarpScan<GradientPairInt64>;

  typename WarpScanT::TempStorage *tmp_storage;
  bst_bin_t gidx_begin;
  bst_bin_t gidx_end;
  bst_target_t n_targets;

  template <typename BinIndexFn>
  __device__ void ScanFeature(GradientPairInt64 const *node_histogram,
                              GradientPairInt64 *scan_result, bst_target_t t,
                              BinIndexFn &&bin_idx_fn) const {
    auto lane_id = static_cast<bst_bin_t>(cuda::ptx::get_sreg_laneid());
    // The forward pass and the backward pass differs in where the bin is read, which is
    // specified by the callback bin_idx_fn(). They write to the same output location.
    GradientPairInt64 warp_aggregate;
    for (auto scan_begin = gidx_begin; scan_begin < gidx_end; scan_begin += dh::WarpThreads()) {
      auto bin_idx = scan_begin + lane_id;
      bool thread_active = bin_idx < gidx_end;
      // Read from histogram: [target][bins]
      auto bin = thread_active ? node_histogram[bin_idx_fn(bin_idx)] : GradientPairInt64{};
      if (lane_id == 0) {
        bin += warp_aggregate;
      }
      WarpScanT(*tmp_storage).InclusiveScan(bin, bin, cuda::std::plus{}, warp_aggregate);
      // Required by the warp scan.
      __syncwarp();
      if (thread_active) {
        // Write to scan result: [bins][targets]
        // The layout is changed from target-major to bin-major here.
        scan_result[bin_idx * n_targets + t] = bin;
      }
    }
  }
  // Forward scan pass
  __device__ void Forward(GradientPairInt64 const *node_histogram,
                          common::Span<GradientPairInt64> scan_result, bst_target_t t) const {
    this->ScanFeature(node_histogram, scan_result.data(), t, cuda::std::identity{});
  }
  // Backward scan pass for missing values
  __device__ void Backward(GradientPairInt64 const *node_histogram,
                           common::Span<GradientPairInt64> scan_result, bst_target_t t) const {
    this->ScanFeature(node_histogram, scan_result.data(), t,
                      [&](bst_bin_t bin_idx) { return RevBinIdx(gidx_begin, gidx_end, bin_idx); });
  }

  // One-hot pass for categorical features.
  //
  // For categorical features the two scan-buffer regions are not a forward/backward scan
  // duality; they hold the two independent missing-direction candidates. We write the
  // *non-missing child sum* for each direction:
  //   - region_others: the matching category goes right with missing (missing-right). The
  //     written non-missing child is the left.
  //   - region_match: the matching category goes right without missing (missing-left). The
  //     written non-missing child is the right.
  __device__ void OneHot(GradientPairInt64 const *node_histogram,
                         common::Span<GradientPairInt64> region_others,
                         common::Span<GradientPairInt64> region_match, bst_target_t t) const {
    auto lane_id = static_cast<bst_bin_t>(cuda::ptx::get_sreg_laneid());
    // Feature sum across all bins for this target.
    GradientPairInt64 local{};
    for (auto bin_idx = gidx_begin + lane_id; bin_idx < gidx_end; bin_idx += dh::WarpThreads()) {
      local += node_histogram[bin_idx];
    }
    auto feature_sum = WarpSum(local);
    // Per-bin child sums, written in the bin-major layout: [bins][targets].
    for (auto bin_idx = gidx_begin + lane_id; bin_idx < gidx_end; bin_idx += dh::WarpThreads()) {
      auto bin = node_histogram[bin_idx];
      region_others[bin_idx * n_targets + t] = feature_sum - bin;
      region_match[bin_idx * n_targets + t] = bin;
    }
  }

  __device__ void Partition(GradientPairInt64 const *node_histogram,
                            common::Span<std::size_t const> sorted_idx,
                            common::Span<GradientPairInt64> forward,
                            common::Span<GradientPairInt64> backward, bst_target_t t) const {
    this->ScanFeature(node_histogram, forward.data(), t,
                      [&](bst_bin_t bin_idx) { return sorted_idx[bin_idx]; });
    this->ScanFeature(node_histogram, backward.data(), t, [&](bst_bin_t bin_idx) {
      return sorted_idx[RevBinIdx(gidx_begin, gidx_end, bin_idx)];
    });
  }
};
}  // namespace

// The scan kernel reads from target-major histogram layout and writes the bin-major scan
// buffer. This helps us keep a reference to the bin in the split candidate.
template <std::int32_t kBlockThreads>
__global__ __launch_bounds__(kBlockThreads) void ScanHistogramKernel(
    common::Span<MultiEvaluateSplitInputs const> nodes, MultiEvaluateSplitSharedInputs shared,
    common::Span<std::size_t const> sorted_idx,
    common::Span<common::Span<GradientPairInt64>> outputs) {
  static_assert(kBlockThreads % dh::WarpThreads() == 0);

  constexpr std::int32_t kWarpsPerBlk = kBlockThreads / dh::WarpThreads();
  auto const warp_id_in_blk = static_cast<std::int32_t>(threadIdx.x) / dh::WarpThreads();
  // The warp index across the entire grid
  auto const warp_id = warp_id_in_blk + kWarpsPerBlk * blockIdx.x;
  bst_target_t const n_targets = shared.Targets();
  auto const n_valid_warps = nodes.size() * shared.max_active_feature * n_targets;

  if (warp_id >= n_valid_warps) {
    return;
  }

  auto [nidx_in_set, fidx_in_set, target_idx] =
      linalg::UnravelIndex(warp_id, nodes.size(), shared.max_active_feature, n_targets);
  auto const &node = nodes[nidx_in_set];
  auto out = outputs[nidx_in_set];
  // This node might have a smaller number of sampled features.
  if (fidx_in_set >= node.feature_set.size()) {
    return;
  }
  auto fidx = node.feature_set[fidx_in_set];
  // The histogram is full, regardless of whether a feature is sampled.
  bst_bin_t gidx_begin = shared.feature_segments[fidx];
  bst_bin_t gidx_end = shared.feature_segments[fidx + 1];

  using AgentT = ScanHistogramAgent;
  __shared__ typename AgentT::WarpScanT::TempStorage tmp_storage[kWarpsPerBlk];
  ScanHistogramAgent agent{&tmp_storage[warp_id_in_blk], gidx_begin, gidx_end, n_targets};
  auto t_hist =
      node.histogram.subspan(shared.n_total_bins_per_tar * target_idx, shared.n_total_bins_per_tar);

  if (shared.IsCategorical(fidx)) {
    auto first = out.subspan(0, node.histogram.size());
    auto second = out.subspan(node.histogram.size(), node.histogram.size());
    auto n_bins = gidx_end - gidx_begin;
    if (common::UseOneHot(n_bins, shared.param.max_cat_to_onehot)) {
      // Both regions are always required (independent missing-directions), so `one_pass`
      // does not apply to categorical features.
      agent.OneHot(t_hist.data(), first, second, target_idx);
    } else {
      auto node_sorted_idx = sorted_idx.subspan(nidx_in_set * shared.n_total_bins_per_tar,
                                                shared.n_total_bins_per_tar);
      agent.Partition(t_hist.data(), node_sorted_idx, first, second, target_idx);
    }
    return;
  }

  if (shared.one_pass != MultiEvaluateSplitSharedInputs::kBackward) {
    auto forward = out.subspan(0, node.histogram.size());
    agent.Forward(t_hist.data(), forward, target_idx);
  }
  // TODO(jiamingy): Skip the backward pass if there's no missing value.
  if (shared.one_pass != MultiEvaluateSplitSharedInputs::kForward) {
    auto backward = out.subspan(node.histogram.size(), node.histogram.size());
    agent.Backward(t_hist.data(), backward, target_idx);
  }
}

namespace {
struct QuantizedGradientSum {
  GradientPairInt64 const *values;
  common::Span<GradientQuantiser const> roundings;

  [[nodiscard]] XGBOOST_DEVICE std::size_t Size() const { return roundings.size(); }
  [[nodiscard]] XGBOOST_DEVICE GradientPairPrecise operator()(std::size_t t) const {
    return roundings.data()[t].ToFloatingPoint(values[t]);
  }
};

struct QuantizedGradientDifference {
  GradientPairInt64 const *parent;
  GradientPairInt64 const *child;
  common::Span<GradientQuantiser const> roundings;

  [[nodiscard]] XGBOOST_DEVICE std::size_t Size() const { return roundings.size(); }
  [[nodiscard]] XGBOOST_DEVICE GradientPairPrecise operator()(std::size_t t) const {
    auto parent_t = roundings.data()[t].ToFloatingPoint(parent[t]);
    auto child_t = roundings.data()[t].ToFloatingPoint(child[t]);
    return parent_t - child_t;
  }
};

static_assert(std::is_trivially_copyable_v<QuantizedGradientSum>);
static_assert(std::is_trivially_copyable_v<QuantizedGradientDifference>);

struct EvaluateSplitAgent {
  using ArgMaxT = cub::KeyValuePair<std::uint32_t, double>;
  using MaxReduceT = cub::WarpReduce<ArgMaxT>;

  typename MaxReduceT::TempStorage *temp_storage;
  bst_feature_t fidx;
  TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator;

  // Calculate the split gain for one bin. `child_scan` has the bin-major layout
  // [bins][targets] and stores the non-missing child sum.
  template <DefaultDirection d_dir>
  static __device__ double ComputeGain(
      MultiEvaluateSplitInputs const &node, MultiEvaluateSplitSharedInputs const &shared,
      common::Span<GradientPairInt64 const> child_scan, bst_bin_t bin_idx, bst_target_t n_targets,
      bst_feature_t fidx, TreeEvaluator::SplitEvaluator<GPUTrainingParam> const &evaluator) {
    auto offset = bin_idx * n_targets;
    auto child_values = child_scan.subspan(offset, n_targets);
    QuantizedGradientSum child{child_values.data(), shared.roundings};
    QuantizedGradientDifference sibling{node.parent_sum.data(), child_values.data(),
                                        shared.roundings};
    if constexpr (d_dir == kRightDir) {
      return evaluator.CalcSplitGain(shared.param, node.nidx, fidx, child, sibling);
    } else {
      return evaluator.CalcSplitGain(shared.param, node.nidx, fidx, sibling, child);
    }
  }

  template <DefaultDirection d_dir>
  __device__ void Numerical(MultiEvaluateSplitInputs const &node,
                            MultiEvaluateSplitSharedInputs const &shared,
                            common::Span<GradientPairInt64 const> node_scan,
                            MultiSplitCandidate *best_split) {
    // Calculate split gain for each bin
    auto n_targets = shared.Targets();
    auto lane_id = static_cast<bst_bin_t>(cuda::ptx::get_sreg_laneid());

    bst_bin_t gidx_begin = shared.feature_segments[fidx];
    bst_bin_t gidx_end = shared.feature_segments[fidx + 1];

    for (auto scan_begin = gidx_begin; scan_begin < gidx_end; scan_begin += dh::WarpThreads()) {
      auto bin_idx = scan_begin + lane_id;
      bool thread_active = bin_idx < gidx_end;

      auto constexpr kNullGain = -std::numeric_limits<double>::infinity();
      double gain = thread_active ? ComputeGain<d_dir>(node, shared, node_scan, bin_idx, n_targets,
                                                       fidx, evaluator)
                                  : kNullGain;

      auto best = MaxReduceT(*temp_storage).Reduce({threadIdx.x, gain}, cub::ArgMax{});
      auto best_thread = __shfl_sync(dh::WarpFullMask(), best.key, 0);

      if (threadIdx.x == best_thread && !isinf(gain)) {
        // Update
        bst_bin_t split_gidx = bin_idx;
        if (d_dir == kLeftDir) {
          split_gidx = RevBinIdx(gidx_begin, gidx_end, bin_idx);
        }
        float fvalue;
        if (d_dir == kRightDir) {
          fvalue = shared.feature_values[split_gidx];
        } else {
          if (split_gidx == gidx_begin) {
            fvalue = -std::numeric_limits<float>::infinity();
          } else {
            fvalue = shared.feature_values[split_gidx - 1];
          }
        }
        // Scan result layout: [bins][targets] - all targets for this bin are contiguous
        // bin_idx is the global bin index
        auto scan_bin_offset = bin_idx * n_targets;
        auto scan_bin = node_scan.subspan(scan_bin_offset, n_targets);
        // Missing values go to right in the forward pass, go to left in the backward pass.
        best_split->Update(gain, d_dir, fvalue, fidx, scan_bin, false);
      }

      __syncwarp();
    }
  }

  // One-hot split for a categorical feature. `region` holds the non-missing child sum for
  // one missing-direction (see ScanHistogramAgent::OneHot). Unlike the numerical backward
  // pass there is no reverse indexing, and the split value is the category id stored at the
  // bin. `d_dir` is the default (missing) direction: kRightDir pairs with `region_others`,
  // kLeftDir pairs with `region_match`.
  template <DefaultDirection d_dir>
  __device__ void OneHot(MultiEvaluateSplitInputs const &node,
                         MultiEvaluateSplitSharedInputs const &shared,
                         common::Span<GradientPairInt64 const> region,
                         MultiSplitCandidate *best_split) {
    auto n_targets = shared.Targets();
    auto lane_id = static_cast<bst_bin_t>(cuda::ptx::get_sreg_laneid());

    bst_bin_t gidx_begin = shared.feature_segments[fidx];
    bst_bin_t gidx_end = shared.feature_segments[fidx + 1];

    for (auto scan_begin = gidx_begin; scan_begin < gidx_end; scan_begin += dh::WarpThreads()) {
      auto bin_idx = scan_begin + lane_id;
      bool thread_active = bin_idx < gidx_end;

      auto constexpr kNullGain = -std::numeric_limits<double>::infinity();
      double gain = thread_active ? ComputeGain<d_dir>(node, shared, region, bin_idx, n_targets,
                                                       fidx, evaluator)
                                  : kNullGain;

      auto best = MaxReduceT(*temp_storage).Reduce({threadIdx.x, gain}, cub::ArgMax{});
      auto best_thread = __shfl_sync(dh::WarpFullMask(), best.key, 0);

      if (threadIdx.x == best_thread && !isinf(gain)) {
        // The split value is the category id (the cut value at this bin).
        float fvalue = shared.feature_values[bin_idx];
        // The scan_bin is directionless, `d_dir` is the carrier of the missing
        // direction. We use it to recover the bin value and sibling value later.
        auto scan_bin = region.subspan(bin_idx * n_targets, n_targets);
        best_split->Update(gain, d_dir, fvalue, fidx, scan_bin, /*cat=*/true);
      }

      __syncwarp();
    }
  }

  template <DefaultDirection d_dir>
  __device__ void Partition(MultiEvaluateSplitInputs const &node,
                            MultiEvaluateSplitSharedInputs const &shared,
                            common::Span<GradientPairInt64 const> node_scan,
                            MultiSplitCandidate *best_split) {
    auto n_targets = shared.Targets();
    auto lane_id = static_cast<bst_bin_t>(cuda::ptx::get_sreg_laneid());

    bst_bin_t gidx_begin = shared.feature_segments[fidx];
    bst_bin_t gidx_end = shared.feature_segments[fidx + 1];
    bst_bin_t n_bins_feature = gidx_end - gidx_begin;
    bst_bin_t n_bins = std::min(shared.param.max_cat_threshold, n_bins_feature);
    if (n_bins <= 1) {
      return;
    }

    // A partition must leave at least one category on each side.
    for (bst_bin_t scan_begin = gidx_begin; scan_begin < gidx_begin + n_bins - 1;
         scan_begin += dh::WarpThreads()) {
      auto bin_idx = scan_begin + lane_id;
      bool thread_active = bin_idx < gidx_begin + n_bins - 1;

      auto constexpr kNullGain = -std::numeric_limits<double>::infinity();
      double gain = thread_active ? ComputeGain<d_dir>(node, shared, node_scan, bin_idx, n_targets,
                                                       fidx, evaluator)
                                  : kNullGain;

      auto best = MaxReduceT(*temp_storage).Reduce({threadIdx.x, gain}, cub::ArgMax{});
      auto best_thread = __shfl_sync(dh::WarpFullMask(), best.key, 0);

      if (threadIdx.x == best_thread && !isinf(gain)) {
        auto scan_offset = bin_idx - gidx_begin;
        auto thresh = d_dir == kLeftDir ? scan_offset
                                        : static_cast<bst_bin_t>(n_bins_feature - scan_offset - 1);
        auto scan_bin = node_scan.subspan(bin_idx * n_targets, n_targets);
        // The forward scan selects a right-child prefix with missing values on the
        // left. The backward scan selects a left-child suffix with missing on the right.
        best_split->UpdateCat(gain, d_dir, static_cast<bst_cat_t>(thresh), fidx, scan_bin);
      }

      __syncwarp();
    }
  }
};
}  // namespace

// Find the best split based on the scan result
//
// The scan buffer has a bin-major layout.
template <std::int32_t kBlockThreads>
__global__ __launch_bounds__(kBlockThreads) void EvaluateSplitsKernel(
    common::Span<MultiEvaluateSplitInputs const> nodes, MultiEvaluateSplitSharedInputs shared,
    common::Span<common::Span<GradientPairInt64>> bin_scans,
    TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
    common::Span<MultiSplitCandidate> out_candidates) {
  static_assert(kBlockThreads % dh::WarpThreads() == 0);

  constexpr std::int32_t kWarpsPerBlk = kBlockThreads / dh::WarpThreads();
  auto const warp_id_in_blk = static_cast<std::int32_t>(threadIdx.x) / dh::WarpThreads();
  // The warp index across the entire grid
  auto const warp_id = warp_id_in_blk + kWarpsPerBlk * blockIdx.x;
  auto const n_valid_warps = nodes.size() * shared.max_active_feature;

  if (warp_id >= n_valid_warps) {
    return;
  }

  using AgentT = EvaluateSplitAgent;
  __shared__ typename AgentT::MaxReduceT::TempStorage temp_storage[kWarpsPerBlk];

  const auto nidx = warp_id / shared.max_active_feature;
  auto const &node = nodes[nidx];

  bst_feature_t fidx_in_set = warp_id - (nidx * shared.max_active_feature);
  // This node might have a smaller number of sampled features.
  if (fidx_in_set >= node.feature_set.size()) {
    return;
  }
  auto fidx = node.feature_set[fidx_in_set];
  AgentT agent{&temp_storage[warp_id_in_blk], fidx, evaluator};
  // The number of candidates is allocated using active features
  auto candidate_idx = nidx * shared.max_active_feature + fidx_in_set;

  if (shared.IsCategorical(fidx)) {
    auto first = bin_scans[nidx].subspan(0, node.histogram.size());
    auto second = bin_scans[nidx].subspan(node.histogram.size(), node.histogram.size());
    auto n_bins = shared.feature_segments[fidx + 1] - shared.feature_segments[fidx];
    if (common::UseOneHot(n_bins, shared.param.max_cat_to_onehot)) {
      // Both regions are always evaluated (independent missing-directions), so `one_pass`
      // does not apply to categorical features.
      agent.template OneHot<kRightDir>(node, shared, first, &out_candidates[candidate_idx]);
      agent.template OneHot<kLeftDir>(node, shared, second, &out_candidates[candidate_idx]);
    } else {
      agent.template Partition<kLeftDir>(node, shared, first, &out_candidates[candidate_idx]);
      agent.template Partition<kRightDir>(node, shared, second, &out_candidates[candidate_idx]);
    }
    return;
  }

  if (shared.one_pass != MultiEvaluateSplitSharedInputs::kBackward) {
    auto forward = bin_scans[nidx].subspan(0, node.histogram.size());
    agent.template Numerical<kRightDir>(node, shared, forward, &out_candidates[candidate_idx]);
  }
  if (shared.one_pass != MultiEvaluateSplitSharedInputs::kForward) {
    auto backward = bin_scans[nidx].subspan(node.histogram.size(), node.histogram.size());
    agent.template Numerical<kLeftDir>(node, shared, backward, &out_candidates[candidate_idx]);
  }
}

void MultiHistEvaluator::Reset(Context const *ctx,
                               common::Span<std::uint32_t const> feature_segments,
                               common::Span<FeatureType const> feature_types,
                               TrainParam const &param) {
  CHECK_GT(feature_segments.size(), 0);
  auto n_features = static_cast<bst_feature_t>(feature_segments.size() - 1);
  this->tree_evaluator_ = TreeEvaluator{param, n_features, ctx->Device()};
  this->need_sort_histogram_ = false;
  if (feature_types.empty()) {
    return;
  }

  auto feature_it = thrust::make_counting_iterator<bst_feature_t>(0);
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

[[nodiscard]] MultiExpandEntry MultiHistEvaluator::EvaluateSingleSplit(
    Context const *ctx, MultiEvaluateSplitInputs const &input,
    MultiEvaluateSplitSharedInputs const &shared_inputs) {
  dh::device_vector<MultiEvaluateSplitInputs> inputs{input};
  dh::device_vector<MultiExpandEntry> outputs(1);

  auto d_outputs = dh::ToSpan(outputs);
  this->EvaluateSplits(ctx, dh::ToSpan(inputs), shared_inputs, input.nidx, d_outputs);

  return outputs[0];
}

namespace {
// Sort histogram based on projected score, see CPU implementation for details.
void SortHistogram(Context const *ctx, MultiEvaluateSplitSharedInputs const &shared_inputs,
                   common::Span<MultiEvaluateSplitInputs const> d_inputs,
                   TreeEvaluator::SplitEvaluator<GPUTrainingParam> evaluator,
                   dh::device_vector<std::size_t> *p_sorted_idx) {
  auto &sorted_idx = *p_sorted_idx;
  auto n_nodes = d_inputs.size();

  auto bins_per_tar = shared_inputs.n_total_bins_per_tar;
  std::size_t total_bins = static_cast<std::size_t>(n_nodes) * bins_per_tar;

  sorted_idx.resize(total_bins);
  auto d_sorted_idx = dh::ToSpan(sorted_idx);

  using SortKey = cuda::std::tuple<std::size_t, bst_feature_t, double>;
  dh::device_vector<SortKey> keys(total_bins);

  auto cnt_it = thrust::make_counting_iterator(0ul);
  thrust::transform(
      ctx->CUDACtx()->CTP(), cnt_it, cnt_it + total_bins, keys.begin(),
      [=] XGBOOST_DEVICE(std::size_t i) {
        d_sorted_idx[i] = i % bins_per_tar;  // segmented iota
        auto [nidx_in_set, bin_idx] = linalg::UnravelIndex(i, n_nodes, bins_per_tar);
        auto fidx =
            static_cast<bst_feature_t>(dh::SegmentId(shared_inputs.feature_segments, bin_idx));
        auto n_cats =
            shared_inputs.feature_segments[fidx + 1] - shared_inputs.feature_segments[fidx];
        bool is_partition = shared_inputs.IsCategorical(fidx) &&
                            !common::UseOneHot(n_cats, shared_inputs.param.max_cat_to_onehot);

        double sc = 0.0;
        if (!is_partition) {
          return cuda::std::make_tuple(nidx_in_set, fidx, sc);
        }

        auto const &node = d_inputs[nidx_in_set];
        for (bst_target_t t = 0; t < shared_inputs.Targets(); ++t) {
          auto quantizer = shared_inputs.roundings[t];
          auto target_hist = node.histogram.subspan(t * bins_per_tar, bins_per_tar);
          auto child_sum = quantizer.ToFloatingPoint(target_hist[bin_idx]);
          auto parent_sum = quantizer.ToFloatingPoint(node.parent_sum[t]);
          auto child_weight = evaluator.CalcWeightCat(shared_inputs.param, child_sum);
          auto parent_weight = evaluator.CalcWeight(node.nidx, shared_inputs.param, parent_sum);
          sc += child_weight * parent_weight;
        }
        return cuda::std::make_tuple(nidx_in_set, fidx, sc);
      });
  thrust::stable_sort_by_key(ctx->CUDACtx()->CTP(), keys.begin(), keys.end(), sorted_idx.begin(),
                             [] XGBOOST_DEVICE(SortKey const &l, SortKey const &r) {
                               // nidx_in_set
                               if (cuda::std::get<0>(l) != cuda::std::get<0>(r)) {
                                 return cuda::std::get<0>(l) < cuda::std::get<0>(r);
                               }
                               // fidx
                               if (cuda::std::get<1>(l) != cuda::std::get<1>(r)) {
                                 return cuda::std::get<1>(l) < cuda::std::get<1>(r);
                               }
                               // score
                               return cuda::std::get<2>(l) < cuda::std::get<2>(r);
                             });
}
}  // namespace

void MultiHistEvaluator::EvaluateSplits(Context const *ctx,
                                        common::Span<MultiEvaluateSplitInputs const> d_inputs,
                                        MultiEvaluateSplitSharedInputs const &shared_inputs,
                                        bst_node_t max_nidx,
                                        common::Span<MultiExpandEntry> out_splits) {
  auto n_targets = shared_inputs.Targets();
  auto evaluator = this->GetEvaluator();
  CHECK_GT(n_targets, 0);
  CHECK_GE(shared_inputs.n_total_bins_per_tar, 1);
  auto n_features = shared_inputs.max_active_feature;
  CHECK_GE(n_features, 1);
  CHECK_LT(n_features, shared_inputs.feature_segments.size());

  std::uint32_t n_nodes = d_inputs.size();
  CHECK_EQ(n_nodes, out_splits.size());

  if (n_nodes == 0) {
    return;
  }

  // Allocate weight and split sum storage on demand for the maximum node ID being evaluated.
  this->AllocNodeWeight(max_nidx, n_targets);
  this->split_sums_.Alloc(max_nidx, n_targets);

  // Calculate total scan buffer size needed for all nodes
  auto node_hist_size = shared_inputs.n_total_bins_per_tar * n_targets;
  std::size_t total_hist_size = node_hist_size * n_nodes;

  // Scan the histograms. One for forward and the other for backward.
  // Since there's only store op on the scan buffer, no need to initialize it.
  this->scan_buffer_.resize(total_hist_size * 2);

  // Create spans for each node's scan results
  std::vector<common::Span<GradientPairInt64>> h_scans(n_nodes);
  for (decltype(n_nodes) nidx_in_set = 0; nidx_in_set < n_nodes; ++nidx_in_set) {
    h_scans[nidx_in_set] = dh::ToSpan(this->scan_buffer_)
                               .subspan(nidx_in_set * node_hist_size * 2, node_hist_size * 2);
  }
  dh::device_vector<common::Span<GradientPairInt64>> scans(h_scans);

  if (shared_inputs.cat_storage_size > 0) {
    this->AllocNodeCats(max_nidx, shared_inputs.cat_storage_size);
  }

  // The values are node-local bin indices. Sort by (node, feature, score)
  dh::device_vector<std::size_t> sorted_idx;
  if (this->need_sort_histogram_) {
    SortHistogram(ctx, shared_inputs, d_inputs, evaluator, &sorted_idx);
  }
  auto d_sorted_idx = common::Span<std::size_t const>{dh::ToSpan(sorted_idx)};

  // Launch histogram scan kernel, each warp handles one target of one feature of one node.
  {
    std::uint32_t constexpr kBlockThreads = 512;
    constexpr std::int32_t kWarpsPerBlk = kBlockThreads / dh::WarpThreads();
    auto n_warps = n_nodes * n_targets * n_features;
    auto n_blocks = common::DivRoundUp(n_warps, kWarpsPerBlk);
    dh::LaunchKernel{n_blocks, kBlockThreads}(  // NOLINT
        ScanHistogramKernel<kBlockThreads>, d_inputs, shared_inputs, d_sorted_idx,
        dh::ToSpan(scans));
  }

  // Launch split evaluation kernel
  dh::device_vector<MultiSplitCandidate> d_splits(n_nodes * n_features);
  {
    std::uint32_t constexpr kBlockThreads = 512;
    constexpr std::int32_t kWarpsPerBlk = kBlockThreads / dh::WarpThreads();
    auto n_warps = n_nodes * n_features;
    auto n_blocks = common::DivRoundUp(n_warps, kWarpsPerBlk);
    dh::LaunchKernel{n_blocks, kBlockThreads, 0, ctx->CUDACtx()->Stream()}(  // NOLINT
        EvaluateSplitsKernel<kBlockThreads>, d_inputs, shared_inputs, dh::ToSpan(scans), evaluator,
        dh::ToSpan(d_splits));
  }

  // Find best split for each node
  auto d_weights = this->GetNodeWeights(n_targets);
  auto d_split_sums = this->split_sums_.View();
  auto d_split_cats = dh::ToSpan(this->split_cats_);
  auto node_cat_storage_size = this->node_cat_storage_size_;
  auto s_d_splits = dh::ToSpan(d_splits);

  // Process results for each node
  // Find best splits among all features for all nodes
  auto key_it = dh::MakeIndexTransformIter([=] XGBOOST_DEVICE(std::size_t i) {
    // Returns nidx_in_set
    return i / n_features;
  });
  dh::device_vector<MultiSplitCandidate> best_splits(out_splits.size());
  thrust::reduce_by_key(
      ctx->CUDACtx()->CTP(), key_it, key_it + s_d_splits.size(), dh::tcbegin(s_d_splits),
      thrust::make_discard_iterator(), best_splits.begin(), std::equal_to{},
      [=] XGBOOST_DEVICE(MultiSplitCandidate const &lhs, MultiSplitCandidate const &rhs) {
        if (lhs.loss_chg > rhs.loss_chg) {
          return lhs;
        }
        if (rhs.loss_chg > lhs.loss_chg) {
          return rhs;
        }
        return lhs.findex <= rhs.findex ? lhs : rhs;
      });
  auto d_best_splits = dh::ToSpan(best_splits);

  dh::LaunchN(n_nodes, ctx->CUDACtx()->Stream(), [=] __device__(std::size_t nidx_in_set) {
    auto input = d_inputs[nidx_in_set];
    MultiSplitCandidate best_split = d_best_splits[nidx_in_set];
    common::Span<common::CatBitField::value_type> node_cats;
    if (!d_split_cats.empty()) {
      node_cats = d_split_cats.subspan(input.nidx * node_cat_storage_size, node_cat_storage_size);
      // Clear the bits, we use it to store the best candidate.
      for (auto &node_cat : node_cats) {
        node_cat = 0;
      }
    }

    // The root weight is required even when no valid split is found. In particular,
    // max_cat_threshold=1 does not enumerate any partition candidate.
    bst_node_t nidx = input.nidx;
    auto base_weight = d_weights.Base(nidx);
    auto roundings = shared_inputs.roundings;
    QuantizedGradientSum parent_sum{input.parent_sum.data(), roundings};
    double parent_gain = evaluator.CalcGain(nidx, shared_inputs.param, parent_sum);
    double parent_hess = 0;
    for (bst_target_t t = 0; t < n_targets; ++t) {
      auto g = roundings[t].ToFloatingPoint(input.parent_sum[t]);
      base_weight[t] = evaluator.CalcWeight(nidx, shared_inputs.param, g);
      parent_hess += g.GetHess();
    }

    if (best_split.child_sum.empty()) {
      // Invalid split
      out_splits[nidx_in_set] = {nidx, input.depth, best_split, base_weight};
      out_splits[nidx_in_set].UpdateHessian(parent_hess, 0.0);
      return;
    }

    if (best_split.is_cat) {
      common::CatBitField cats{node_cats};
      if (!isnan(best_split.fvalue)) {  // OHE
        cats.Set(common::AsCat(best_split.fvalue));
      } else {  // Partition
        auto fidx = best_split.findex;
        auto f_begin = shared_inputs.feature_segments[fidx];
        auto n_bins =
            shared_inputs.feature_segments[fidx + 1] - shared_inputs.feature_segments[fidx];
        auto node_sorted_idx = d_sorted_idx.subspan(
            nidx_in_set * shared_inputs.n_total_bins_per_tar, shared_inputs.n_total_bins_per_tar);
        auto f_sorted_idx = node_sorted_idx.subspan(f_begin, n_bins);
        bst_bin_t partition = best_split.dir == kLeftDir
                                  ? static_cast<bst_bin_t>(best_split.thresh + 1)
                                  : static_cast<bst_bin_t>(best_split.thresh);
        KERNEL_CHECK(partition > 0);
        for (bst_bin_t i = 0; i < partition; ++i) {
          auto cat = shared_inputs.feature_values[f_sorted_idx[i]];
          cats.Set(common::AsCat(cat));
        }
      }
    }

    // Calculate weights for this node using the actual node id for persistent storage
    auto left_weight = d_weights.Left(nidx);
    auto right_weight = d_weights.Right(nidx);

    auto split_sum = best_split.child_sum;

    // Copy split sum to persistent buffer for loss-guide grow policy support.
    // The child_sum span in best_split points to scan_buffer_ which gets reused,
    // so we store it persistently indexed by node id.
    auto split_sum_dest = GetNodeSumImpl(d_split_sums, nidx, n_targets);

    double left_hess = 0, right_hess = 0;  // Sum of child hessians across all targets

    for (bst_target_t t = 0; t < n_targets; ++t) {
      auto quantizer = roundings[t];
      auto sibling_sum = input.parent_sum[t] - split_sum[t];

      split_sum_dest[t] = split_sum[t];

      // Left/right weights
      GradientPairPrecise lg, rg;
      if (best_split.dir == kRightDir) {
        // forward pass, split_sum is the left sum
        lg = quantizer.ToFloatingPoint(split_sum[t]);
        left_weight[t] = evaluator.CalcWeight(nidx, shared_inputs.param, lg);
        rg = quantizer.ToFloatingPoint(sibling_sum);
        right_weight[t] = evaluator.CalcWeight(nidx, shared_inputs.param, rg);
      } else {
        // backward pass, split_sum is the right sum
        rg = quantizer.ToFloatingPoint(split_sum[t]);
        right_weight[t] = evaluator.CalcWeight(nidx, shared_inputs.param, rg);
        lg = quantizer.ToFloatingPoint(sibling_sum);
        left_weight[t] = evaluator.CalcWeight(nidx, shared_inputs.param, lg);
      }

      left_hess += lg.GetHess();
      right_hess += rg.GetHess();
    }

    // Set up the output entry with spans pointing to persistent weight storage
    out_splits[nidx_in_set] = {nidx, input.depth, best_split, base_weight};
    out_splits[nidx_in_set].split.loss_chg -= parent_gain;
    out_splits[nidx_in_set].UpdateHessian(left_hess, right_hess);
  });
}

void MultiHistEvaluator::ApplyTreeSplit(Context const *ctx, RegTree const *p_tree,
                                        common::Span<MultiExpandEntry const> d_candidates,
                                        bst_target_t n_targets) {
  // Assign the node sums here, for the next evaluate split call.
  auto mt_tree = MultiTargetTreeView{ctx->Device(), false, p_tree};
  auto max_in_it = dh::MakeIndexTransformIter([=] __device__(std::size_t i) -> bst_node_t {
    return std::max(mt_tree.LeftChild(d_candidates[i].nidx),
                    mt_tree.RightChild(d_candidates[i].nidx));
  });
  auto max_node = thrust::reduce(
      ctx->CUDACtx()->CTP(), max_in_it, max_in_it + d_candidates.size(), 0,
      [=] XGBOOST_DEVICE(bst_node_t l, bst_node_t r) { return cuda::std::max(l, r); });
  this->AllocNodeSum(max_node, n_targets);

  auto node_sums = this->node_sums_.View();
  // Use the internal split sums buffer instead of candidate.split.child_sum . It may be
  // stale in loss-guide grow policy (entries can remain in priority queue across
  // evaluation rounds).
  auto split_sums = this->split_sums_.View();

  dh::LaunchN(n_targets * d_candidates.size(), ctx->CUDACtx()->Stream(),
              [=] XGBOOST_DEVICE(std::size_t i) {
                auto get_node_sum = [&](bst_node_t nidx) {
                  return GetNodeSumImpl(node_sums, nidx, n_targets);
                };
                auto nidx_in_set = i / n_targets;
                auto t = i % n_targets;

                auto const &candidate = d_candidates[nidx_in_set];
                auto const &best_split = candidate.split;

                auto parent_sum = get_node_sum(candidate.nidx);
                // Look up split sum from persistent buffer by node id.
                // Use split_targets for indexing since that's what was used during storage.
                auto split_sum = GetNodeSumImpl(split_sums, candidate.nidx, n_targets);
                auto left_sum = get_node_sum(mt_tree.LeftChild(candidate.nidx));
                auto right_sum = get_node_sum(mt_tree.RightChild(candidate.nidx));

                auto split_sum_t = split_sum[t];
                auto sibling_sum = parent_sum[t] - split_sum_t;
                if (best_split.dir == kRightDir) {
                  // forward pass, node_sum is the left sum
                  left_sum[t] = split_sum_t;
                  right_sum[t] = sibling_sum;
                } else {
                  // backward pass, node_sum is the right sum
                  right_sum[t] = split_sum_t;
                  left_sum[t] = sibling_sum;
                }
              });
}
}  // namespace xgboost::tree::cuda_impl
