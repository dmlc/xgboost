/**
 * Copyright 2025, XGBoost contributors
 */
#include <thrust/reduce.h>  // for reduce_by_key, reduce

#include <cub/block/block_scan.cuh>  // for BlockScan
#include <cub/util_type.cuh>         // for KeyValuePair
#include <cub/warp/warp_reduce.cuh>  // for WarpReduce
#include <cuda/std/functional>       // for identity
#include <vector>                    // for vector

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
                              BinIndexFn &&bin_idx_fn) {
    auto lane_id = dh::LaneId();
    // The forward pass and the backward pass differs in where the bin is read, which is
    // specified by the callback bin_idx_fn(). They write to the same output location.
    GradientPairInt64 warp_aggregate;
    for (bst_bin_t scan_begin = gidx_begin; scan_begin < gidx_end;
         scan_begin += dh::WarpThreads()) {
      auto bin_idx = scan_begin + lane_id;
      bool thread_active = bin_idx < gidx_end;
      auto bin =
          thread_active ? node_histogram[bin_idx_fn(bin_idx) * n_targets + t] : GradientPairInt64{};
      if (lane_id == 0) {
        bin += warp_aggregate;
      }
      WarpScanT(*tmp_storage).InclusiveScan(bin, bin, cuda::std::plus{}, warp_aggregate);
      // Required by the warp scan.
      __syncwarp();
      if (thread_active) {
        scan_result[bin_idx * n_targets + t] = bin;
      }
    }
  }
  // Forward scan pass
  __device__ void Forward(GradientPairInt64 const *node_histogram,
                          common::Span<GradientPairInt64> scan_result, bst_target_t t) {
    this->ScanFeature(node_histogram, scan_result.data(), t, cuda::std::identity{});
  }
  // Backward scan pass for missing values
  __device__ void Backward(GradientPairInt64 const *node_histogram,
                           common::Span<GradientPairInt64> scan_result, bst_target_t t) {
    this->ScanFeature(node_histogram, scan_result.data(), t,
                      [&](bst_bin_t bin_idx) { return RevBinIdx(gidx_begin, gidx_end, bin_idx); });
  }
};
}  // namespace

template <std::int32_t kBlockThreads>
__global__ __launch_bounds__(kBlockThreads) void ScanHistogramKernel(
    common::Span<MultiEvaluateSplitInputs const> nodes, MultiEvaluateSplitSharedInputs shared,
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
  // The current histogram layout has consecutive targets, which results in excessive
  // (non-coalesced) memory access for the evaluation kernels.
  auto [nidx_in_set, fidx, target_idx] =
      linalg::UnravelIndex(warp_id, nodes.size(), shared.max_active_feature, n_targets);
  auto const &node = nodes[nidx_in_set];
  auto out = outputs[nidx_in_set];

  bst_bin_t gidx_begin = shared.feature_segments[fidx];
  bst_bin_t gidx_end = shared.feature_segments[fidx + 1];

  using AgentT = ScanHistogramAgent;
  __shared__ typename AgentT::WarpScanT::TempStorage tmp_storage[kWarpsPerBlk];
  ScanHistogramAgent agent{&tmp_storage[warp_id_in_blk], gidx_begin, gidx_end, n_targets};

  if (shared.one_pass != MultiEvaluateSplitSharedInputs::kBackward) {
    auto forward = out.subspan(0, node.histogram.size());
    agent.Forward(node.histogram.data(), forward, target_idx);
  }
  // TODO(jiamingy): Skip the backward pass if there's no missing value.
  if (shared.one_pass != MultiEvaluateSplitSharedInputs::kForward) {
    auto backward = out.subspan(node.histogram.size(), node.histogram.size());
    agent.Backward(node.histogram.data(), backward, target_idx);
  }
}

namespace {
struct EvaluateSplitAgent {
  using ArgMaxT = cub::KeyValuePair<std::uint32_t, double>;
  using MaxReduceT = cub::WarpReduce<ArgMaxT>;
  using SumReduceT = cub::WarpReduce<GradientPairInt64>;

  struct TempStorage {
    typename MaxReduceT::TempStorage max_reduce;
    typename SumReduceT::TempStorage sum_reduce;
  } *temp_storage;
  bst_feature_t fidx;

  template <std::int32_t d_step>
  __device__ void Numerical(MultiEvaluateSplitInputs const &node,
                            MultiEvaluateSplitSharedInputs const &shared,
                            common::Span<GradientPairInt64 const> node_scan,
                            MultiSplitCandidate *best_split) {
    static_assert(d_step == +1 || d_step == -1, "Invalid step.");
    // Calculate split gain for each bin
    auto n_targets = shared.Targets();
    auto roundings = shared.roundings;
    auto lane_id = dh::LaneId();

    bst_bin_t gidx_begin = shared.feature_segments[fidx];
    bst_bin_t gidx_end = shared.feature_segments[fidx + 1];

    for (bst_bin_t scan_begin = gidx_begin; scan_begin < gidx_end;
         scan_begin += dh::WarpThreads()) {
      auto bin_idx = scan_begin + lane_id;
      bool thread_active = bin_idx < gidx_end;

      auto constexpr kNullGain = -std::numeric_limits<double>::infinity();
      double gain = thread_active ? 0 : kNullGain;

      if (thread_active) {
        auto scan_bin = node_scan.subspan(bin_idx * n_targets, n_targets);
        for (bst_target_t t = 0; t < n_targets; ++t) {
          auto pg = roundings[t].ToFloatingPoint(node.parent_sum[t]);
          // left
          auto left_sum = roundings[t].ToFloatingPoint(scan_bin[t]);
          auto lw_t =
              ::xgboost::tree::CalcWeight(shared.param, left_sum.GetGrad(), left_sum.GetHess());
          // right
          auto right_sum = pg - left_sum;
          auto rw_t =
              ::xgboost::tree::CalcWeight(shared.param, right_sum.GetGrad(), right_sum.GetHess());

          gain += -lw_t * ThresholdL1(left_sum.GetGrad(), shared.param.reg_alpha);
          gain += -rw_t * ThresholdL1(right_sum.GetGrad(), shared.param.reg_alpha);
        }
      }

      auto best = MaxReduceT(temp_storage->max_reduce).Reduce({threadIdx.x, gain}, cub::ArgMax{});
      auto best_thread = __shfl_sync(0xffffffff, best.key, 0);

      if (threadIdx.x == best_thread && !isinf(gain)) {
        // Update
        bst_bin_t split_gidx = bin_idx;
        if (d_step == -1) {
          split_gidx = RevBinIdx(gidx_begin, gidx_end, bin_idx);
        }
        float min_fvalue = shared.min_values[fidx];
        float fvalue;
        if (d_step == +1) {
          fvalue = shared.feature_values[split_gidx];
        } else {
          if (split_gidx == gidx_begin) {
            fvalue = min_fvalue;
          } else {
            fvalue = shared.feature_values[split_gidx - 1];
          }
        }
        auto scan_bin = node_scan.subspan(bin_idx * n_targets, n_targets);
        // Missing values go to right in the forward pass, go to left in the backward pass.
        best_split->Update(gain, d_step == 1 ? kRightDir : kLeftDir, fvalue, fidx, scan_bin, false,
                           shared.param, shared.roundings);
      }

      __syncwarp();
    }
  }
};
}  // namespace

// Find the best split based on the scan result
template <std::int32_t kBlockThreads>
__global__ __launch_bounds__(kBlockThreads) void EvaluateSplitsKernel(
    common::Span<MultiEvaluateSplitInputs const> nodes, MultiEvaluateSplitSharedInputs shared,
    common::Span<common::Span<GradientPairInt64>> bin_scans,
    common::Span<MultiSplitCandidate> out_candidates) {
  constexpr std::int32_t kWarpsPerBlk = kBlockThreads / dh::WarpThreads();
  auto const warp_id_in_blk = static_cast<std::int32_t>(threadIdx.x) / dh::WarpThreads();
  // The warp index across the entire grid
  auto const warp_id = warp_id_in_blk + kWarpsPerBlk * blockIdx.x;
  auto const n_valid_warps = nodes.size() * shared.max_active_feature;

  if (warp_id >= n_valid_warps) {
    return;
  }

  using AgentT = EvaluateSplitAgent;
  __shared__ typename AgentT::TempStorage temp_storage[kWarpsPerBlk];

  const auto nidx = warp_id / shared.max_active_feature;
  bst_feature_t fidx = warp_id % shared.max_active_feature;
  AgentT agent{&temp_storage[warp_id_in_blk], fidx};

  auto n_targets = shared.Targets();
  auto candidate_idx = nidx * shared.max_active_feature + fidx;
  auto d_nodes = nodes.data();

  if (shared.one_pass != MultiEvaluateSplitSharedInputs::kBackward) {
    auto forward = bin_scans[nidx].subspan(0, d_nodes[nidx].histogram.size());
    agent.template Numerical<+1>(d_nodes[nidx], shared, forward, &out_candidates[candidate_idx]);
  }
  if (shared.one_pass != MultiEvaluateSplitSharedInputs::kForward) {
    auto backward =
        bin_scans[nidx].subspan(d_nodes[nidx].histogram.size(), d_nodes[nidx].histogram.size());
    agent.template Numerical<-1>(d_nodes[nidx], shared, backward, &out_candidates[candidate_idx]);
  }
}

[[nodiscard]] MultiExpandEntry MultiHistEvaluator::EvaluateSingleSplit(
    Context const *ctx, MultiEvaluateSplitInputs const &input,
    MultiEvaluateSplitSharedInputs const &shared_inputs) {
  dh::device_vector<MultiEvaluateSplitInputs> inputs{input};
  dh::device_vector<MultiExpandEntry> outputs(1);

  auto d_outputs = dh::ToSpan(outputs);
  this->EvaluateSplits(ctx, dh::ToSpan(inputs), shared_inputs, d_outputs);

  // The `EvaluateSplits` apply eta for leaf nodes only, we need to apply it for the base
  // weight.
  auto n_targets = shared_inputs.Targets();
  dh::LaunchN(n_targets, ctx->CUDACtx()->Stream(), [=] XGBOOST_DEVICE(std::size_t t) {
    auto weight = d_outputs[0].base_weight;
    if (weight.empty()) {
      return;
    }
    weight[t] *= shared_inputs.param.learning_rate;
  });

  return outputs[0];
}

void MultiHistEvaluator::EvaluateSplits(Context const *ctx,
                                        common::Span<MultiEvaluateSplitInputs const> d_inputs,
                                        MultiEvaluateSplitSharedInputs const &shared_inputs,
                                        common::Span<MultiExpandEntry> out_splits) {
  auto n_targets = shared_inputs.Targets();
  auto n_bins_per_feat_tar = shared_inputs.n_bins_per_feat_tar;
  CHECK_GE(n_bins_per_feat_tar, 1);
  auto n_features = shared_inputs.max_active_feature;
  CHECK_GE(n_features, 1);
  CHECK_LT(n_features, shared_inputs.feature_segments.size());

  std::uint32_t n_nodes = d_inputs.size();
  CHECK_EQ(n_nodes, out_splits.size());

  if (n_nodes == 0) {
    return;
  }

  // Calculate total scan buffer size needed for all nodes
  auto node_hist_size = n_targets * n_features * n_bins_per_feat_tar;
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

  // Launch histogram scan kernel, each warp handles one target of one feature of one node.
  {
    std::uint32_t constexpr kBlockThreads = 512;
    constexpr std::int32_t kWarpsPerBlk = kBlockThreads / dh::WarpThreads();
    auto n_warps = n_nodes * n_targets * n_features;
    auto n_blocks = common::DivRoundUp(n_warps, kWarpsPerBlk);
    dh::LaunchKernel{n_blocks, kBlockThreads}(  // NOLINT
        ScanHistogramKernel<kBlockThreads>, d_inputs, shared_inputs, dh::ToSpan(scans));
  }

  // Launch split evaluation kernel
  dh::device_vector<MultiSplitCandidate> d_splits(n_nodes * n_features);
  {
    std::uint32_t constexpr kBlockThreads = 512;
    constexpr std::int32_t kWarpsPerBlk = kBlockThreads / dh::WarpThreads();
    auto n_warps = n_nodes * n_features;
    auto n_blocks = common::DivRoundUp(n_warps, kWarpsPerBlk);
    dh::LaunchKernel{n_blocks, kBlockThreads, 0, ctx->CUDACtx()->Stream()}(  // NOLINT
        EvaluateSplitsKernel<kBlockThreads>, d_inputs, shared_inputs, dh::ToSpan(scans),
        dh::ToSpan(d_splits));
  }

  // Find best split for each node
  // * 3 because of base, left, right weights.
  this->weights_.resize(n_nodes * n_targets * 3);
  auto d_weights = dh::ToSpan(this->weights_);

  dh::CachingDeviceUVector<float> d_parent_gains(n_nodes);
  auto s_parent_gains = dh::ToSpan(d_parent_gains);
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
        return lhs.loss_chg > rhs.loss_chg ? lhs : rhs;
      });
  auto d_best_splits = dh::ToSpan(best_splits);

  dh::LaunchN(n_nodes, ctx->CUDACtx()->Stream(), [=] __device__(std::size_t nidx_in_set) {
    auto input = d_inputs[nidx_in_set];
    MultiSplitCandidate best_split = d_best_splits[nidx_in_set];
    if (best_split.child_sum.empty()) {
      // Invalid split
      out_splits[nidx_in_set] = {};
      return;
    }

    // Calculate weights for this node
    auto base_weight = d_weights.subspan(nidx_in_set * n_targets * 3, n_targets);
    auto left_weight = d_weights.subspan(nidx_in_set * n_targets * 3 + n_targets, n_targets);
    auto right_weight = d_weights.subspan(nidx_in_set * n_targets * 3 + n_targets * 2, n_targets);

    auto d_roundings = shared_inputs.roundings;
    auto node_sum = best_split.child_sum;

    float parent_gain = 0;
    for (bst_target_t t = 0; t < n_targets; ++t) {
      auto quantizer = d_roundings[t];
      auto sibling_sum = input.parent_sum[t] - node_sum[t];
      auto sum = node_sum[t] + sibling_sum;
      auto g = quantizer.ToFloatingPoint(sum);

      base_weight[t] = CalcWeight(shared_inputs.param, g.GetGrad(), g.GetHess());
      parent_gain += -base_weight[t] * ThresholdL1(g.GetGrad(), shared_inputs.param.reg_alpha);
    }
    s_parent_gains[nidx_in_set] = parent_gain;

    bool l = true, r = true;
    GradientPairPrecise lg_fst, rg_fst;
    auto eta = shared_inputs.param.learning_rate;
    for (bst_target_t t = 0; t < n_targets; ++t) {
      auto quantizer = d_roundings[t];
      auto sibling_sum = input.parent_sum[t] - node_sum[t];

      l = l && (node_sum[t].GetQuantisedHess() == 0);
      r = r && (sibling_sum.GetQuantisedHess() == 0);

      GradientPairPrecise lg, rg;
      if (best_split.dir == kRightDir) {
        // forward pass, node_sum is the left sum
        lg = quantizer.ToFloatingPoint(node_sum[t]);
        left_weight[t] = CalcWeight(shared_inputs.param, lg.GetGrad(), lg.GetHess()) * eta;
        rg = quantizer.ToFloatingPoint(sibling_sum);
        right_weight[t] = CalcWeight(shared_inputs.param, rg.GetGrad(), rg.GetHess()) * eta;
      } else {
        // backward pass, node_sum is the right sum
        rg = quantizer.ToFloatingPoint(node_sum[t]);
        right_weight[t] = CalcWeight(shared_inputs.param, rg.GetGrad(), rg.GetHess()) * eta;
        lg = quantizer.ToFloatingPoint(sibling_sum);
        left_weight[t] = CalcWeight(shared_inputs.param, lg.GetGrad(), lg.GetHess()) * eta;
      }

      if (t == 0) {
        lg_fst = lg;
        rg_fst = rg;
      }
    }

    // Set up the output entry
    out_splits[nidx_in_set] = {input.nidx,  input.depth, best_split,
                               base_weight, left_weight, right_weight};
    out_splits[nidx_in_set].split.loss_chg -= parent_gain;
    out_splits[nidx_in_set].UpdateFirstHessian(lg_fst, rg_fst);

    if (l || r) {
      out_splits[nidx_in_set] = {};
    }
  });
}

void MultiHistEvaluator::ApplyTreeSplit(Context const *ctx, RegTree const *p_tree,
                                        common::Span<MultiExpandEntry const> d_candidates,
                                        bst_target_t n_targets) {
  // Assign the node sums here, for the next evaluate split call.
  auto mt_tree = MultiTargetTreeView{ctx->Device(), p_tree};
  auto max_in_it = dh::MakeIndexTransformIter([=] __device__(std::size_t i) -> bst_node_t {
    return std::max(mt_tree.LeftChild(d_candidates[i].nidx),
                    mt_tree.RightChild(d_candidates[i].nidx));
  });
  auto max_node = thrust::reduce(
      ctx->CUDACtx()->CTP(), max_in_it, max_in_it + d_candidates.size(), 0,
      [=] XGBOOST_DEVICE(bst_node_t l, bst_node_t r) { return cuda::std::max(l, r); });
  this->AllocNodeSum(max_node, n_targets);

  auto node_sums = dh::ToSpan(this->node_sums_);

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
                // The child sum is a pointer to the scan buffer in this evaluator. Copy
                // the data into the node sum buffer before the next evaluation call.
                auto node_sum = best_split.child_sum;
                auto left_sum = get_node_sum(mt_tree.LeftChild(candidate.nidx));
                auto right_sum = get_node_sum(mt_tree.RightChild(candidate.nidx));

                auto sibling_sum = parent_sum[t] - node_sum[t];
                if (best_split.dir == kRightDir) {
                  // forward pass, node_sum is the left sum
                  left_sum[t] = node_sum[t];
                  right_sum[t] = sibling_sum;
                } else {
                  // backward pass, node_sum is the right sum
                  right_sum[t] = node_sum[t];
                  left_sum[t] = sibling_sum;
                }
              });
}

std::ostream &DebugPrintHistogram(std::ostream &os, common::Span<GradientPairInt64 const> node_hist,
                                  common::Span<GradientQuantiser const> roundings,
                                  bst_target_t n_targets) {
  std::vector<GradientQuantiser> h_roundings;
  thrust::copy(dh::tcbegin(roundings), dh::tcend(roundings), std::back_inserter(h_roundings));
  dh::CopyDeviceSpanToVector(&h_roundings, roundings);

  std::vector<GradientPairInt64> h_node_hist(node_hist.size());
  dh::CopyDeviceSpanToVector(&h_node_hist, node_hist);
  for (bst_target_t t = 0; t < n_targets; ++t) {
    os << "Target:" << t << std::endl;
    for (std::size_t i = t; i < h_node_hist.size() / n_targets; i += n_targets) {
      os << h_roundings[t].ToFloatingPoint(h_node_hist[i]) << ", ";
    }
    os << std::endl;
  }
  return os;
}
}  // namespace xgboost::tree::cuda_impl
