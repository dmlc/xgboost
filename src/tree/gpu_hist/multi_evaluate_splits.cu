/**
 * Copyright 2025, XGBoost contributors
 */
#include <cub/block/block_scan.cuh>  // for BlockScan
#include <cub/util_type.cuh>         // for KeyValuePair
#include <cub/warp/warp_reduce.cuh>  // for WarpReduce
#include <vector>                    // for vector

#include "../../common/cuda_context.cuh"
#include "../updater_gpu_common.cuh"  // for SumCallbackOp
#include "multi_evaluate_splits.cuh"  // for MultiEvalauteSplitInputs, MultiEvaluateSplitSharedInputs
#include "quantiser.cuh"              // for GradientQuantiser
#include "xgboost/base.h"             // for GradientPairInt64
#include "xgboost/span.h"             // for Span

namespace xgboost::tree::cuda_impl {
namespace {
__device__ bst_bin_t RevBinIdx(bst_bin_t gidx_end, bst_bin_t bin_idx) {
  return gidx_end - bin_idx - 1;
}

// Scan the histogram in 2 dim for all nodes
// Each block for one feature and one target
template <std::int32_t kBlockThreads>
struct ScanHistogramAgent {
  using BlockScanT = cub::BlockScan<GradientPairInt64, kBlockThreads>;

  typename BlockScanT::TempStorage *tmp_storage;
  bst_bin_t gidx_begin;
  bst_bin_t gidx_end;
  bst_target_t n_targets;

  template <typename BinIndexFn>
  __device__ void ScanFeature(common::Span<GradientPairInt64 const> node_histogram,
                              common::Span<GradientPairInt64> scan_result, bst_target_t t,
                              BinIndexFn &&bin_idx_fn) {
    SumCallbackOp<GradientPairInt64> prefix_op;
    // The forward pass and the backward pass differs in where the bin is read, which is
    // specified by the callback bin_idx_fn(). They write to the same output location.
    for (bst_bin_t scan_begin = gidx_begin; scan_begin < gidx_end; scan_begin += kBlockThreads) {
      auto bin_idx = scan_begin + threadIdx.x;
      bool thread_active = bin_idx < gidx_end;
      auto bin =
          thread_active ? node_histogram[bin_idx_fn(bin_idx) * n_targets + t] : GradientPairInt64{};
      BlockScanT(*tmp_storage).InclusiveScan(bin, bin, cuda::std::plus{}, prefix_op);
      if (thread_active) {
        scan_result[bin_idx * n_targets + t] = bin;
      }

      // Required by the block scan.
      __syncthreads();
    }
  }
  // Forward scan pass
  __device__ void Forward(common::Span<GradientPairInt64 const> node_histogram,
                          common::Span<GradientPairInt64> scan_result, bst_target_t t) {
    this->ScanFeature(node_histogram, scan_result, t, cuda::std::identity{});
  }
  // Backward scan pass for missing values
  __device__ void Backward(common::Span<GradientPairInt64 const> node_histogram,
                           common::Span<GradientPairInt64> scan_result, bst_target_t t) {
    this->ScanFeature(node_histogram, scan_result, t,
                      [&](bst_bin_t bin_idx) { return RevBinIdx(gidx_end, bin_idx); });
  }
};
}  // namespace

template <std::int32_t kBlockThreads>
__global__ __launch_bounds__(kBlockThreads) void ScanHistogramKernel(
    common::Span<MultiEvaluateSplitInputs const> nodes, MultiEvaluateSplitSharedInputs shared,
    common::Span<common::Span<GradientPairInt64>> outputs) {
  auto nidx_in_set = blockIdx.x;

  auto const &node = nodes[nidx_in_set];
  auto out = outputs[nidx_in_set];

  auto fidx = blockIdx.y;
  auto t = blockIdx.z;

  bst_bin_t gidx_begin = shared.feature_segments[fidx];
  bst_bin_t gidx_end = shared.feature_segments[fidx + 1];
  bst_target_t n_targets = shared.Targets();

  using AgentT = ScanHistogramAgent<kBlockThreads>;
  __shared__ typename AgentT::BlockScanT::TempStorage tmp_storage;
  ScanHistogramAgent<kBlockThreads> agent{&tmp_storage, gidx_begin, gidx_end, n_targets};

  if (shared.one_pass != MultiEvaluateSplitSharedInputs::kBackward) {
    auto forward = out.subspan(0, node.histogram.size());
    agent.Forward(node.histogram, forward, t);
  }
  // TODO(jiamingy): Skip the backward pass if there's no missing value.
  if (shared.one_pass != MultiEvaluateSplitSharedInputs::kForward) {
    auto backward = out.subspan(node.histogram.size(), node.histogram.size());
    agent.Backward(node.histogram, backward, t);
  }
}

namespace {
template <std::int32_t kBlockThreads>
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
                            common::Span<GradientPairInt64 const> f_scan,
                            MultiSplitCandidate *best_split) {
    static_assert(d_step == +1 || d_step == -1, "Invalid step.");
    // Calculate split gain for each bin
    auto n_targets = shared.Targets();
    auto roundings = shared.roundings;

    bst_bin_t gidx_begin = shared.feature_segments[fidx];
    bst_bin_t gidx_end = shared.feature_segments[fidx + 1];

    for (bst_bin_t scan_begin = gidx_begin; scan_begin < gidx_end; scan_begin += kBlockThreads) {
      auto bin_idx = scan_begin + threadIdx.x;
      bool thread_active = bin_idx < gidx_end;

      auto constexpr kNullGain = -std::numeric_limits<double>::infinity();
      double gain = thread_active ? 0 : kNullGain;

      if (thread_active) {
        auto scan_bin = f_scan.subspan(bin_idx * n_targets, n_targets);
        for (bst_target_t t = 0; t < n_targets; ++t) {
          auto pg = roundings[t].ToFloatingPoint(node.parent_sum[t]);
          // left
          SPAN_LT(t, scan_bin.size());
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
          split_gidx = RevBinIdx(gidx_end, bin_idx);
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
        auto scan_bin = f_scan.subspan(bin_idx * n_targets, n_targets);
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
// Only a single node is working at the moment
template <std::int32_t kBlockThreads>
__global__ __launch_bounds__(kBlockThreads) void EvaluateSplitsKernel(
    common::Span<MultiEvaluateSplitInputs const> nodes, MultiEvaluateSplitSharedInputs shared,
    common::Span<common::Span<GradientPairInt64>> bin_scans,
    common::Span<MultiSplitCandidate> out_candidates) {
  using AgentT = EvaluateSplitAgent<kBlockThreads>;
  __shared__ typename AgentT::TempStorage temp_storage;

  auto fidx = blockIdx.x;
  EvaluateSplitAgent<kBlockThreads> agent{&temp_storage, blockIdx.x};

  auto n_targets = shared.Targets();
  // The number of bins in a feature
  auto f_hist_size =
      (shared.feature_segments[fidx + 1] - shared.feature_segments[fidx]) * n_targets;
  // TODO(jiamingy): Support more than a single node

  if (shared.one_pass != MultiEvaluateSplitSharedInputs::kBackward) {
    auto forward = bin_scans[0].subspan(0, nodes[0].histogram.size());
    auto f_scan = forward.subspan(shared.feature_segments[fidx] * n_targets, f_hist_size);
    agent.template Numerical<+1>(nodes[0], shared, f_scan, &out_candidates[fidx]);
  }
  if (shared.one_pass != MultiEvaluateSplitSharedInputs::kForward) {
    auto backward = bin_scans[0].subspan(nodes[0].histogram.size(), nodes[0].histogram.size());
    auto f_scan = backward.subspan(shared.feature_segments[fidx] * n_targets, f_hist_size);
    agent.template Numerical<-1>(nodes[0], shared, f_scan, &out_candidates[fidx]);
  }
}

[[nodiscard]] MultiExpandEntry MultiHistEvaluator::EvaluateSingleSplit(
    Context const *ctx, MultiEvaluateSplitInputs input,
    MultiEvaluateSplitSharedInputs shared_inputs) {
  auto n_targets = shared_inputs.Targets();
  CHECK_GE(n_targets, 2);
  auto n_bins_per_feat_tar = shared_inputs.n_bins_per_feat_tar;
  CHECK_GE(n_bins_per_feat_tar, 1);
  auto n_features = shared_inputs.Features();
  CHECK_GE(n_features, 1);

  dh::device_vector<MultiEvaluateSplitInputs> inputs{input};

  // Scan the histograms. One for forward and the other for backward.
  this->scan_buffer_.resize(input.histogram.size() * 2);
  thrust::fill(ctx->CUDACtx()->CTP(), this->scan_buffer_.begin(), this->scan_buffer_.end(),
               GradientPairInt64{});
  dh::device_vector<common::Span<GradientPairInt64>> scans{dh::ToSpan(this->scan_buffer_)};
  std::uint32_t n_nodes = 1;
  dim3 grid{n_nodes, n_features, n_targets};
  std::uint32_t constexpr kBlockThreads = 32;
  dh::LaunchKernel{grid, kBlockThreads}(  // NOLINT
      ScanHistogramKernel<kBlockThreads>, dh::ToSpan(inputs), shared_inputs, dh::ToSpan(scans));

  dh::device_vector<MultiSplitCandidate> d_splits(n_features);
  dh::LaunchKernel{n_features, kBlockThreads, 0, ctx->CUDACtx()->Stream()}(  // NOLINT
      EvaluateSplitsKernel<kBlockThreads>, dh::ToSpan(inputs), shared_inputs, dh::ToSpan(scans),
      dh::ToSpan(d_splits));

  auto best_split = thrust::reduce(
      ctx->CUDACtx()->CTP(), d_splits.cbegin(), d_splits.cend(), MultiSplitCandidate{},
      [] XGBOOST_DEVICE(MultiSplitCandidate const &lhs, MultiSplitCandidate const &rhs)
          -> MultiSplitCandidate { return lhs.loss_chg > rhs.loss_chg ? lhs : rhs; });

  if (best_split.node_sum.empty()) {
    return {};
  }

  // Calculate leaf weights from gradient sum
  this->weights_.resize(n_targets * 3);
  auto d_weights = dh::ToSpan(this->weights_);
  auto base_weight = d_weights.subspan(0, n_targets);
  auto left_weight = d_weights.subspan(n_targets, n_targets);
  auto right_weight = d_weights.subspan(n_targets * 2, n_targets);

  dh::CachingDeviceUVector<float> d_parent_gain(1);
  dh::CachingDeviceUVector<std::int32_t> sum_zero(2);

  auto s_pg = dh::ToSpan(d_parent_gain);
  auto s_sum_zero = dh::ToSpan(sum_zero);

  dh::LaunchN(inputs.size(), ctx->CUDACtx()->Stream(), [=] __device__(std::size_t i) {
    auto d_roundings = shared_inputs.roundings;
    // the data inside the split candidates references the scan result.
    auto node_sum = best_split.node_sum;

    float parent_gain = 0;
    for (bst_target_t t = 0; t < n_targets; ++t) {
      auto quantizer = d_roundings[t];
      auto sibling_sum = input.parent_sum[t] - node_sum[t];
      auto sum = node_sum[t] + sibling_sum;
      auto g = quantizer.ToFloatingPoint(sum);

      base_weight[t] = CalcWeight(shared_inputs.param, g.GetGrad(), g.GetHess());
      parent_gain += -base_weight[t] * ThresholdL1(g.GetGrad(), shared_inputs.param.reg_alpha);
    }
    s_pg[0] = parent_gain;

    bool l = true, r = true;
    for (bst_target_t t = 0; t < n_targets; ++t) {
      auto quantizer = d_roundings[t];
      auto sibling_sum = input.parent_sum[t] - node_sum[t];

      l = l && (node_sum[t].GetQuantisedHess() - .0 == .0);
      r = r && (sibling_sum.GetQuantisedHess() - .0 == .0);

      if (best_split.dir == kRightDir) {
        // forward pass, node_sum is the left sum
        auto lg = quantizer.ToFloatingPoint(node_sum[t]);
        left_weight[t] = CalcWeight(shared_inputs.param, lg.GetGrad(), lg.GetHess());
        auto rg = quantizer.ToFloatingPoint(sibling_sum);
        right_weight[t] = CalcWeight(shared_inputs.param, rg.GetGrad(), rg.GetHess());
      } else {
        // backward pass, node_sum is the right sum
        auto rg = quantizer.ToFloatingPoint(node_sum[t]);
        right_weight[t] = CalcWeight(shared_inputs.param, rg.GetGrad(), rg.GetHess());
        auto lg = quantizer.ToFloatingPoint(sibling_sum);
        left_weight[t] = CalcWeight(shared_inputs.param, lg.GetGrad(), lg.GetHess());
      }

      s_sum_zero[0] = l;
      s_sum_zero[1] = r;
    }
  });
  // Copy the result back to the host.
  float parent_gain = 0;
  dh::safe_cuda(cudaMemcpyAsync(&parent_gain, d_parent_gain.data(), sizeof(parent_gain),
                                cudaMemcpyDefault, ctx->CUDACtx()->Stream()));
  best_split.loss_chg -= parent_gain;

  std::vector<std::int32_t> h_sum_zero(s_sum_zero.size());
  dh::safe_cuda(cudaMemcpyAsync(h_sum_zero.data(), s_sum_zero.data(), s_sum_zero.size_bytes(),
                                cudaMemcpyDefault, ctx->CUDACtx()->Stream()));
  if (h_sum_zero[0] || h_sum_zero[1]) {
    return {};
  }

  MultiExpandEntry entry{input.nidx,  input.depth, best_split,
                         base_weight, left_weight, right_weight};
  return entry;
}

void DebugPrintHistogram(common::Span<GradientPairInt64 const> node_hist,
                         common::Span<GradientQuantiser const> roundings, bst_target_t n_targets) {
  std::vector<GradientQuantiser> h_roundings;
  thrust::copy(dh::tcbegin(roundings), dh::tcend(roundings), std::back_inserter(h_roundings));
  dh::CopyDeviceSpanToVector(&h_roundings, roundings);

  std::vector<GradientPairInt64> h_node_hist(node_hist.size());
  dh::CopyDeviceSpanToVector(&h_node_hist, node_hist);
  for (bst_target_t t = 0; t < n_targets; ++t) {
    std::cout << "target:" << t << std::endl;
    for (std::size_t i = t; i < h_node_hist.size() / n_targets; i += n_targets) {
      std::cout << h_roundings[t].ToFloatingPoint(h_node_hist[i]) << ", ";
    }
    std::cout << std::endl;
  }
}
}  // namespace xgboost::tree::cuda_impl
